"""确定性离散事件模拟器。

确定性来源：
- 到达时间由确定性强度函数（基础速率 + 峰值倍率）的反函数解析生成，
  不使用任何随机数；
- 事务时长用黄金比例低差异序列在离散分布上取值，混合充分且可复现；
- 事件堆以 (时间, 类型序号, 单调序号) 排序，耗时几乎相同的并发到达
  按固定规则打破平局。

调度策略（每次连接释放时）：
1. 先丢弃队首已超时的请求；
2. 跳过已达到租户并发上限的服务；
3. 保留容量：服务已占用连接低于其保留数时优先；未保留部分对所有
   服务开放，但非保留占用总量不得超过 (容量 - 保留总量)；
4. 其余按加权公平排队：选择 served/weight 最小的服务，服务内 FIFO。
"""

import heapq
from collections import Counter, defaultdict, deque

_GOLDEN = 0.6180339887498949


def arrival_times(rate, horizon, peak_multiplier=1.0, peak_fraction=0.0):
    """按确定性强度函数生成到达时刻列表。

    强度 lambda(t) 在峰值段为 rate*peak_multiplier，其余段缩放使全程
    平均速率恰为 rate。第 k 个到达时刻 t 满足积分强度 = k + 0.5。
    """
    if rate <= 0.0 or horizon <= 0.0:
        return []
    peak_fraction = min(max(peak_fraction, 0.0), 1.0)
    if peak_multiplier < 1.0:
        peak_multiplier = 1.0
    peak_end = horizon * peak_fraction
    if peak_fraction > 0.0 and peak_multiplier > 1.0:
        off = (1.0 - peak_fraction * peak_multiplier) / (1.0 - peak_fraction)
        off = max(off, 0.0)
    else:
        peak_end = 0.0
        off = 1.0
    peak_rate = rate * peak_multiplier
    off_rate = rate * off
    total = rate * horizon
    count = int(total)
    times = []
    for k in range(count):
        target = k + 0.5
        peak_cum = peak_rate * peak_end
        if target <= peak_cum and peak_rate > 0.0:
            times.append(target / peak_rate)
        elif off_rate > 0.0:
            times.append(peak_end + (target - peak_cum) / off_rate)
        else:
            times.append(peak_end)
    return times


def duration_for(durations, index):
    """用黄金比例序列在第 index 个请求上确定性地取一个时长。"""
    total_w = sum(w for _, w in durations)
    if total_w <= 0.0:
        return 0.0
    u = (index * _GOLDEN) % 1.0
    acc = 0.0
    for dur, w in durations:
        acc += w / total_w
        if u < acc:
            return dur
    return durations[-1][0]


def _percentile(sorted_values, q):
    """确定性线性插值分位数。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


class _Request(object):
    __slots__ = ("service", "tenant", "arrive", "duration", "max_wait")

    def __init__(self, service, tenant, arrive, duration, max_wait):
        self.service = service
        self.tenant = tenant
        self.arrive = arrive
        self.duration = duration
        self.max_wait = max_wait


def simulate_instance(requests, capacity, weights, reserved, tenant_caps, horizon):
    """在单个实例上模拟。

    requests:     _Request 列表（任意顺序，内部按到达时间入堆）。
    capacity:     该实例分到的池大小。
    weights:      {服务名: 调度权重}。
    reserved:     {服务名: 该实例上保留的连接数}。
    tenant_caps:  {租户: 该实例上的并发上限}，缺省表示不限。
    返回 (per_service_metrics, tenant_occupancy, utilization)。
    """
    events = []
    seq = 0
    for req in requests:
        heapq.heappush(events, (req.arrive, 0, seq, req))
        seq += 1

    queues = defaultdict(deque)
    active = 0
    active_service = Counter()
    active_tenant = Counter()
    served = Counter()
    waits = defaultdict(list)
    timeouts = Counter()
    arrivals = Counter()
    occupancy_area = Counter()
    occupancy_max = Counter()
    busy_area = 0.0
    last_time = 0.0

    total_reserved = sum(reserved.values())

    def advance_clock(now):
        nonlocal last_time, busy_area
        dt = now - last_time
        if dt > 0.0:
            for tenant, cnt in active_tenant.items():
                if cnt:
                    occupancy_area[tenant] += cnt * dt
            busy_area += active * dt
            last_time = now

    def try_start(now):
        nonlocal active
        while active < capacity:
            # 1. 丢弃队首超时请求
            for name in list(queues.keys()):
                q = queues[name]
                while q and now - q[0].arrive > q[0].max_wait:
                    timeouts[q[0].service] += 1
                    q.popleft()
                if not q:
                    del queues[name]
            # 2-4. 选择下一个可启动的服务
            unreserved_busy = sum(
                max(0, active_service[n] - reserved.get(n, 0))
                for n in active_service
            )
            best = None
            best_key = None
            for name, q in queues.items():
                req = q[0]
                cap = tenant_caps.get(req.tenant)
                if cap is not None and active_tenant[req.tenant] >= cap:
                    continue
                if active_service[name] >= reserved.get(name, 0):
                    if unreserved_busy >= capacity - total_reserved:
                        continue
                key = (served[name] / weights.get(name, 1.0), req.arrive, name)
                if best_key is None or key < best_key:
                    best_key = key
                    best = name
            if best is None:
                break
            req = queues[best].popleft()
            if not queues[best]:
                del queues[best]
            served[best] += 1
            active += 1
            active_service[best] += 1
            active_tenant[req.tenant] += 1
            if active_tenant[req.tenant] > occupancy_max[req.tenant]:
                occupancy_max[req.tenant] = active_tenant[req.tenant]
            waits[best].append(now - req.arrive)
            heapq.heappush(
                events, (now + req.duration, 1, seq_holder[0], (best, req.tenant))
            )
            seq_holder[0] += 1

    seq_holder = [seq]

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if now > horizon:
            break
        advance_clock(now)
        if kind == 0:
            arrivals[payload.service] += 1
            queues[payload.service].append(payload)
        else:
            name, tenant = payload
            active -= 1
            active_service[name] -= 1
            active_tenant[tenant] -= 1
        try_start(now)

    # 模拟结束时仍滞留队列的请求按超时计。
    for name, q in queues.items():
        timeouts[name] += len(q)
    advance_clock(horizon)

    per_service = {}
    names = set(arrivals) | set(waits) | set(timeouts)
    for name in sorted(names):
        w = sorted(waits[name])
        done = len(w)
        timed_out = timeouts[name]
        total = done + timed_out
        per_service[name] = {
            "arrivals": arrivals[name],
            "completed": done,
            "timeouts": timed_out,
            "timeout_rate": (timed_out / total) if total else 0.0,
            "avg_wait": (sum(w) / done) if done else 0.0,
            "p95_wait": _percentile(w, 0.95),
            "max_wait": w[-1] if w else 0.0,
        }
    occupancy = {
        t: occupancy_area[t] / horizon for t in sorted(occupancy_area)
    }
    max_occupancy = {t: occupancy_max[t] for t in sorted(occupancy_max)}
    utilization = busy_area / (capacity * horizon) if capacity > 0 else 0.0
    return per_service, occupancy, max_occupancy, utilization
