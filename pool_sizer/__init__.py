"""确定性数据库连接池容量规划器。

通过离散事件模拟（而非平均值公式）评估候选池大小，
输出推荐配置与每个服务的预期风险。
"""

from .models import ServiceSpec, InstanceSpec, PlanRequest
from .planner import plan_pool, allocate_connections
from .simulation import simulate_instance

__all__ = [
    "ServiceSpec",
    "InstanceSpec",
    "PlanRequest",
    "plan_pool",
    "allocate_connections",
    "simulate_instance",
]

__version__ = "0.1.0"
