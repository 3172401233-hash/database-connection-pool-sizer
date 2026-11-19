"""输入/输出数据模型。全部使用 Python 3.8 兼容语法。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ServiceSpec:
    """单个服务的负载画像。

    arrival_rate:     平均到达率（请求/秒）。
    durations:        事务时长离散分布，[(时长秒, 权重), ...]。
    max_wait:         请求在队列中允许的最大等待秒数，超时即放弃。
    weight:           队列调度权重（越大越优先被服务）。
    reserved:         为该服务（通常是长事务服务）保留的最小连接数（全局）。
    peak_multiplier:  峰值时段到达率倍率（1.0 表示无峰值）。
    peak_fraction:    峰值时段占模拟时长的比例 (0, 1)。
    """

    name: str
    tenant: str
    arrival_rate: float
    durations: List[Tuple[float, float]]
    max_wait: float
    weight: float = 1.0
    reserved: int = 0
    peak_multiplier: float = 1.0
    peak_fraction: float = 0.0

    def mean_duration(self) -> float:
        total_w = sum(w for _, w in self.durations)
        if total_w <= 0:
            return 0.0
        return sum(d * w for d, w in self.durations) / total_w


@dataclass
class InstanceSpec:
    """数据库实例（主或从）。

    weight:          路由权重，例如主:从 = 1:2 表示 2/3 请求打到从库。
    max_connections: 实例允许的最大连接数（硬上限）。
    """

    name: str
    weight: float = 1.0
    max_connections: int = 100


@dataclass
class PlanRequest:
    """一次容量规划的完整输入。"""

    services: List[ServiceSpec]
    instances: List[InstanceSpec]
    candidates: List[int]                 # 候选总连接数
    tenant_concurrency: Dict[str, int] = field(default_factory=dict)  # 租户并发上限
    horizon: float = 60.0                 # 模拟时长（秒）
    timeout_budget: float = 0.01          # 可接受的单服务超时率上限
