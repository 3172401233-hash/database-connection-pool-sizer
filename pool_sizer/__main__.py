"""命令行入口：读取 JSON 规划请求，输出 JSON 推荐配置。

用法: python -m pool_sizer request.json
"""

import json
import sys

from .models import InstanceSpec, PlanRequest, ServiceSpec
from .planner import plan_pool


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    services = [ServiceSpec(**s) for s in raw["services"]]
    for svc in services:
        svc.durations = [tuple(pair) for pair in svc.durations]
    instances = [InstanceSpec(**i) for i in raw["instances"]]
    return PlanRequest(
        services=services,
        instances=instances,
        candidates=raw["candidates"],
        tenant_concurrency=raw.get("tenant_concurrency", {}),
        horizon=raw.get("horizon", 60.0),
        timeout_budget=raw.get("timeout_budget", 0.01),
    )


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: python -m pool_sizer request.json\n")
        return 2
    plan = plan_pool(_load(argv[1]))
    json.dump(plan, sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
