"""容量规划器：跨实例分配连接，逐候选池大小模拟并给出推荐。"""

from collections import Counter

from .simulation import _Request, arrival_times, duration_for, simulate_instance


def _largest_remainder(total, weights, caps=None):
    """按权重把 total 个整数份额分给各方（最大余数法），可选逐项上限。

    返回 (分配列表, 未能放下的剩余数量)。
    """
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n, max(total, 0)
    caps = caps if caps is not None else [None] * n
    wsum = float(sum(weights))
    if wsum <= 0.0:
        weights = [1.0] * n
        wsum = float(n)
    alloc = [0] * n
    remaining = total
    # 每轮按剩余权重比例分配，余数大的优先，已达上限的跳过。
    while remaining > 0:
        active = [i for i in range(n) if caps[i] is None or alloc[i] < caps[i]]
        if not active:
            break
        aw = sum(weights[i] for i in active)
        if aw <= 0.0:
            break
        raw = {i: remaining * weights[i] / aw for i in active}
        grew = False
        for i in sorted(active, key=lambda i: (-(raw[i] % 1.0), i)):
            if remaining <= 0:
                break
            if caps[i] is not None and alloc[i] >= caps[i]:
                continue
            give = max(1, int(raw[i]))
            give = min(give, remaining)
            if caps[i] is not None:
                give = min(give, caps[i] - alloc[i])
            if give > 0:
                alloc[i] += give
                remaining -= give
                grew = True
        if not grew:
            break
    return alloc, remaining


def allocate_connections(total, instances):
    """把 total 个连接按路由权重分配到实例，受 max_connections 约束。

    返回 {实例名: 连接数}；若实例上限之和放不下 total，返回 None。
    """
    weights = [ins.weight for ins in instances]
    caps = [ins.max_connections for ins in instances]
    alloc, leftover = _largest_remainder(total, weights, caps)
    if leftover > 0:
        return None
    return {ins.name: a for ins, a in zip(instances, alloc)}


def _build_routed_requests(request):
    """为每个服务生成确定性请求流，并按实例权重轮询路由。"""
    weights = [ins.weight for ins in request.instances]
    assigned = [0] * len(request.instances)
    routed = [[] for _ in request.instances]
    for svc in request.services:
        times = arrival_times(
            svc.arrival_rate, request.horizon,
            svc.peak_multiplier, svc.peak_fraction,
        )
        for idx, t in enumerate(times):
            # 加权最少已分配：路由比例严格贴近实例权重且完全确定。
            best = min(
                range(len(request.instances)),
                key=lambda i: (assigned[i] / (weights[i] or 1e-9), i),
            )
            assigned[best] += 1
            routed[best].append(_Request(
                service=svc.name,
                tenant=svc.tenant,
                arrive=t,
                duration=duration_for(svc.durations, idx),
                max_wait=svc.max_wait,
            ))
    return routed


def _merge_service_metrics(all_metrics):
    """把各实例的 per-service 指标合并为全局指标。"""
    merged = {}
    for metrics in all_metrics:
        for name, m in metrics.items():
            slot = merged.setdefault(name, {
                "arrivals": 0, "completed": 0, "timeouts": 0,
                "wait_sum": 0.0, "waits": [],
            })
            slot["arrivals"] += m["arrivals"]
            slot["completed"] += m["completed"]
            slot["timeouts"] += m["timeouts"]
            slot["wait_sum"] += m["avg_wait"] * m["completed"]
            slot["waits"].append(m)
    out = {}
    from .simulation import _percentile
    for name in sorted(merged):
        slot = merged[name]
        done = slot["completed"]
        total = done + slot["timeouts"]
        # p95/max 由各实例分位数加权近似（确定性）。
        p95 = _percentile(sorted(m["p95_wait"] for m in slot["waits"]), 0.95)
        maxw = max((m["max_wait"] for m in slot["waits"]), default=0.0)
        out[name] = {
            "arrivals": slot["arrivals"],
            "completed": done,
            "timeouts": slot["timeouts"],
            "timeout_rate": (slot["timeouts"] / total) if total else 0.0,
            "avg_wait": (slot["wait_sum"] / done) if done else 0.0,
            "p95_wait": p95,
            "max_wait": maxw,
        }
    return out


def _risk_level(timeout_rate, wait_ratio, budget):
    if timeout_rate > budget * 10 or wait_ratio >= 1.0:
        return "high"
    if timeout_rate > budget or wait_ratio >= 0.5:
        return "medium"
    return "low"


def _evaluate(request, total):
    """评估一个候选总连接数，返回候选结果字典。"""
    allocation = allocate_connections(total, request.instances)
    if allocation is None:
        return {
            "total": total,
            "feasible": False,
            "reason": "exceeds_instance_limits",
            "allocation": None,
        }
    routed = _build_routed_requests(request)
    weights = [ins.weight for ins in request.instances]
    svc_by_name = {s.name: s for s in request.services}
    tenants = sorted({s.tenant for s in request.services})

    all_metrics = []
    occupancy = Counter()
    max_occupancy = Counter()
    utilizations = {}
    for i, ins in enumerate(request.instances):
        cap = allocation[ins.name]
        # 每个服务的保留量独立按实例权重拆分。
        reserved_map = {}
        for svc in request.services:
            if svc.reserved > 0:
                shares, _ = _largest_remainder(svc.reserved, weights)
                reserved_map[svc.name] = shares[i]
        tenant_caps = {}
        for tenant in tenants:
            tenant_cap = request.tenant_concurrency.get(tenant)
            if tenant_cap:
                shares, _ = _largest_remainder(tenant_cap, weights)
                tenant_caps[tenant] = max(1, shares[i])
        svc_weights = {s.name: s.weight for s in request.services}
        metrics, occ, occ_max, util = simulate_instance(
            routed[i], cap, svc_weights, reserved_map, tenant_caps,
            request.horizon,
        )
        all_metrics.append(metrics)
        for tenant, val in occ.items():
            occupancy[tenant] += val
        for tenant, val in occ_max.items():
            max_occupancy[tenant] += val
        utilizations[ins.name] = util

    merged = _merge_service_metrics(all_metrics)
    per_service = {}
    feasible = True
    for name, m in merged.items():
        svc = svc_by_name[name]
        wait_ratio = (m["p95_wait"] / svc.max_wait) if svc.max_wait > 0 else 0.0
        level = _risk_level(m["timeout_rate"], wait_ratio, request.timeout_budget)
        if m["timeout_rate"] > request.timeout_budget:
            feasible = False
        per_service[name] = dict(m)
        per_service[name].update({
            "tenant": svc.tenant,
            "wait_ratio": wait_ratio,
            "risk": level,
        })
    return {
        "total": total,
        "feasible": feasible,
        "reason": None if feasible else "timeout_budget_exceeded",
        "allocation": allocation,
        "services": per_service,
        "tenant_occupancy": {t: occupancy[t] for t in sorted(occupancy)},
        "tenant_max_occupancy": {t: max_occupancy[t] for t in sorted(max_occupancy)},
        "utilization": utilizations,
    }


def _score(result):
    """不可行候选之间的确定性评分：越小越好。"""
    services = result.get("services") or {}
    timeout_sum = sum(m["timeout_rate"] for m in services.values())
    wait_sum = sum(m["wait_ratio"] for m in services.values())
    return (timeout_sum, wait_sum, result["total"])


def plan_pool(request):
    """主入口：评估全部候选，返回推荐配置与每个服务的预期风险。"""
    evaluated = [_evaluate(request, c) for c in sorted(set(request.candidates))]
    feasible = [r for r in evaluated if r["feasible"]]
    if feasible:
        # 候选已按总连接数升序，取最小可行。
        recommended = feasible[0]
    else:
        evaluated_with_services = [r for r in evaluated if r.get("services")]
        recommended = (
            min(evaluated_with_services, key=_score)
            if evaluated_with_services else evaluated[-1]
        )
    return {
        "recommended_total": recommended["total"],
        "allocation": recommended["allocation"],
        "feasible": recommended["feasible"],
        "reason": recommended["reason"],
        "services": recommended.get("services", {}),
        "tenant_occupancy": recommended.get("tenant_occupancy", {}),
        "tenant_max_occupancy": recommended.get("tenant_max_occupancy", {}),
        "utilization": recommended.get("utilization", {}),
        "candidates": evaluated,
    }
