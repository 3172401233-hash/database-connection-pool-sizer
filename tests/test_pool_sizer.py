# -*- coding: utf-8 -*-
"""连接池容量规划器测试：全程确定性，不依赖真实数据库或随机数。"""

import json
import unittest

from pool_sizer import (
    InstanceSpec,
    PlanRequest,
    ServiceSpec,
    allocate_connections,
    plan_pool,
)
from pool_sizer.simulation import (
    _Request,
    arrival_times,
    duration_for,
    simulate_instance,
)


def make_request(**overrides):
    base = dict(
        services=[
            ServiceSpec("svc", "tenantA", 10.0, [(0.1, 1.0)], max_wait=0.5),
        ],
        instances=[InstanceSpec("primary", 1.0, 100)],
        candidates=[5, 10, 20],
        tenant_concurrency={"tenantA": 50},
        horizon=20.0,
    )
    base.update(overrides)
    return PlanRequest(**base)


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_output(self):
        req = make_request(
            services=[
                ServiceSpec("a", "t1", 15.0, [(0.05, 3.0), (0.5, 1.0)],
                            max_wait=0.4, peak_multiplier=3.0, peak_fraction=0.2),
                ServiceSpec("b", "t2", 3.0, [(1.0, 1.0)], max_wait=0.8,
                            weight=2.0, reserved=2),
            ],
            instances=[InstanceSpec("p", 1.0, 60), InstanceSpec("r", 2.0, 120)],
            candidates=[8, 16, 32],
            tenant_concurrency={"t1": 20, "t2": 8},
        )
        first = json.dumps(plan_pool(req), sort_keys=True)
        second = json.dumps(plan_pool(req), sort_keys=True)
        self.assertEqual(first, second)

    def test_duration_sequence_reproducible(self):
        dist = [(0.1, 1.0), (0.5, 2.0), (2.0, 1.0)]
        seq1 = [duration_for(dist, i) for i in range(50)]
        seq2 = [duration_for(dist, i) for i in range(50)]
        self.assertEqual(seq1, seq2)
        self.assertTrue(set(seq1) <= {0.1, 0.5, 2.0})

    def test_nearly_identical_concurrent_arrivals_deterministic(self):
        # 几乎相同耗时的并发到达：两次模拟结果必须完全一致。
        def run():
            reqs = [
                _Request("s", "t", arrive=0.0, duration=1.0 + i * 1e-9,
                         max_wait=0.5)
                for i in range(20)
            ]
            return simulate_instance(reqs, 4, {"s": 1.0}, {}, {}, 10.0)

        m1, _, _, _ = run()
        m2, _, _, _ = run()
        self.assertEqual(m1, m2)
        # 容量 4、时长约 1s、20 个请求 0.5s 内必须拿到连接：必有超时。
        self.assertGreater(m1["s"]["timeouts"], 0)


class ArrivalProcessTest(unittest.TestCase):
    def test_sparse_requests(self):
        times = arrival_times(0.5, 10.0)
        self.assertEqual(len(times), 5)
        self.assertEqual(times, [1.0, 3.0, 5.0, 7.0, 9.0])

    def test_peak_profile_concentrates_arrivals(self):
        times = arrival_times(10.0, 10.0, peak_multiplier=3.0, peak_fraction=0.2)
        self.assertEqual(len(times), 100)
        in_peak = sum(1 for t in times if t < 2.0)
        self.assertEqual(in_peak, 60)  # 10*3*2 = 60 个落在峰值段

    def test_zero_rate(self):
        self.assertEqual(arrival_times(0.0, 10.0), [])


class SingleInstanceTest(unittest.TestCase):
    def test_recommends_smallest_feasible(self):
        req = make_request(
            services=[ServiceSpec("svc", "t", 8.0, [(0.2, 1.0)], max_wait=1.0)],
            candidates=[1, 2, 4, 8],
        )
        plan = plan_pool(req)
        # 平均并发需求 = 8 * 0.2 = 1.6，池为 1 必然大量超时。
        by_total = {c["total"]: c for c in plan["candidates"]}
        self.assertFalse(by_total[1]["feasible"])
        self.assertGreater(by_total[1]["services"]["svc"]["timeout_rate"], 0.0)
        self.assertTrue(plan["feasible"])
        self.assertGreaterEqual(plan["recommended_total"], 2)
        self.assertEqual(plan["allocation"], {"primary": plan["recommended_total"]})
        self.assertIn(plan["services"]["svc"]["risk"], ("low", "medium"))

    def test_queue_time_reported(self):
        req = make_request(
            services=[ServiceSpec("svc", "t", 6.0, [(0.5, 1.0)], max_wait=5.0)],
            candidates=[2],
        )
        plan = plan_pool(req)
        svc = plan["candidates"][0]["services"]["svc"]
        self.assertGreater(svc["avg_wait"], 0.0)
        self.assertGreaterEqual(svc["max_wait"], svc["p95_wait"])


class PrimaryReplicaSplitTest(unittest.TestCase):
    def test_allocation_follows_weights(self):
        instances = [InstanceSpec("primary", 1.0, 100),
                     InstanceSpec("replica", 2.0, 200)]
        alloc = allocate_connections(30, instances)
        self.assertEqual(alloc, {"primary": 10, "replica": 20})

    def test_routing_respects_instance_weights(self):
        req = make_request(
            services=[ServiceSpec("svc", "t", 30.0, [(0.1, 1.0)], max_wait=1.0)],
            instances=[InstanceSpec("primary", 1.0, 100),
                       InstanceSpec("replica", 2.0, 200)],
            candidates=[30],
        )
        plan = plan_pool(req)
        self.assertEqual(plan["allocation"], {"primary": 10, "replica": 20})
        # 从库承担 2/3 流量，连接更多但利用率应与主库接近。
        util = plan["utilization"]
        self.assertAlmostEqual(util["primary"], util["replica"], delta=0.15)


class TenantFairnessTest(unittest.TestCase):
    def test_noisy_tenant_cannot_starve_others(self):
        req = make_request(
            services=[
                ServiceSpec("noisy", "big", 200.0, [(0.2, 1.0)], max_wait=0.3),
                ServiceSpec("quiet", "small", 5.0, [(0.1, 1.0)], max_wait=0.5),
            ],
            tenant_concurrency={"big": 8, "small": 8},
            candidates=[24],
            horizon=20.0,
        )
        plan = plan_pool(req)
        svc = plan["services"]
        # 大租户被并发上限与公平调度约束，小租户超时率必须显著更低。
        self.assertLess(svc["quiet"]["timeout_rate"],
                        svc["noisy"]["timeout_rate"])
        self.assertLess(svc["quiet"]["timeout_rate"], 0.05)
        # 大租户全局平均占用不得超过其并发上限。
        self.assertLessEqual(plan["tenant_occupancy"]["big"], 8.0 + 1e-9)
        self.assertLessEqual(plan["tenant_max_occupancy"]["big"], 8)

    def test_reserved_capacity_protects_long_transactions(self):
        def build(reserved):
            # 两个长事务服务竞争 8 个连接：batch 需求 12 并发，
            # long_tx 需求 6 并发，公平调度下各自只能拿到约一半。
            return make_request(
                services=[
                    ServiceSpec("batch", "t1", 6.0, [(2.0, 1.0)],
                                max_wait=2.0),
                    ServiceSpec("long_tx", "t2", 2.0, [(3.0, 1.0)],
                                max_wait=0.5, reserved=reserved),
                ],
                tenant_concurrency={},
                candidates=[8],
                horizon=20.0,
            )
        unprotected = plan_pool(build(0))["services"]["long_tx"]
        protected = plan_pool(build(6))["services"]["long_tx"]
        self.assertGreater(unprotected["timeout_rate"], 0.0)
        # 保留 6 个连接恰好覆盖长事务的稳态并发（2/s * 3s）。
        self.assertLess(protected["timeout_rate"],
                        unprotected["timeout_rate"])
        self.assertEqual(protected["timeout_rate"], 0.0)


class InstanceLimitTest(unittest.TestCase):
    def test_candidate_exceeding_instance_caps_infeasible(self):
        instances = [InstanceSpec("primary", 1.0, 50),
                     InstanceSpec("replica", 1.0, 50)]
        self.assertIsNone(allocate_connections(120, instances))
        req = make_request(instances=instances, candidates=[80, 120])
        plan = plan_pool(req)
        by_total = {c["total"]: c for c in plan["candidates"]}
        self.assertEqual(by_total[120]["reason"], "exceeds_instance_limits")
        self.assertFalse(by_total[120]["feasible"])
        self.assertIsNone(by_total[120]["allocation"])
        # 80 <= 100 上限之和，可以分配。
        self.assertEqual(sum(by_total[80]["allocation"].values()), 80)

    def test_allocation_caps_at_max_connections(self):
        instances = [InstanceSpec("p", 1.0, 10), InstanceSpec("r", 1.0, 100)]
        alloc = allocate_connections(50, instances)
        self.assertEqual(alloc["p"], 10)   # 主库顶到上限
        self.assertEqual(alloc["r"], 40)   # 余量全部给从库


class InfeasibleCandidateTest(unittest.TestCase):
    def test_all_candidates_infeasible_still_recommends_best(self):
        req = make_request(
            services=[ServiceSpec("heavy", "t", 50.0, [(1.0, 1.0)],
                                  max_wait=0.1)],
            candidates=[2, 4],
            horizon=10.0,
        )
        plan = plan_pool(req)
        self.assertFalse(plan["feasible"])
        self.assertEqual(plan["reason"], "timeout_budget_exceeded")
        # 两个候选都不可行时，推荐风险更小的（更大的池）。
        self.assertEqual(plan["recommended_total"], 4)
        self.assertEqual(plan["services"]["heavy"]["risk"], "high")
        self.assertGreater(plan["services"]["heavy"]["timeout_rate"], 0.0)

    def test_zero_capacity_candidate(self):
        req = make_request(candidates=[0, 20])
        plan = plan_pool(req)
        by_total = {c["total"]: c for c in plan["candidates"]}
        self.assertFalse(by_total[0]["feasible"])
        self.assertTrue(plan["feasible"])


class SparseAndBurstTest(unittest.TestCase):
    def test_sparse_service_low_utilization_no_timeout(self):
        req = make_request(
            services=[ServiceSpec("rare", "t", 0.2, [(0.5, 1.0)], max_wait=1.0)],
            candidates=[4],
            horizon=30.0,
        )
        plan = plan_pool(req)
        svc = plan["services"]["rare"]
        self.assertEqual(svc["timeouts"], 0)
        self.assertEqual(svc["arrivals"], 6)
        self.assertLess(plan["utilization"]["primary"], 0.05)

    def test_burst_peak_drives_timeouts(self):
        calm = make_request(
            services=[ServiceSpec("s", "t", 10.0, [(0.3, 1.0)], max_wait=0.2)],
            candidates=[4], horizon=10.0,
        )
        burst = make_request(
            services=[ServiceSpec("s", "t", 10.0, [(0.3, 1.0)], max_wait=0.2,
                                  peak_multiplier=4.0, peak_fraction=0.25)],
            candidates=[4], horizon=10.0,
        )
        calm_rate = plan_pool(calm)["services"]["s"]["timeout_rate"]
        burst_rate = plan_pool(burst)["services"]["s"]["timeout_rate"]
        self.assertGreater(burst_rate, calm_rate)


if __name__ == "__main__":
    unittest.main()
