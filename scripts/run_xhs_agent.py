from __future__ import annotations

import argparse
import asyncio
import json

from app.services.dashscope_client import DashScopeClient
from app.services.xhs_agent import XhsAgentService


async def _main() -> None:
    parser = argparse.ArgumentParser(description='Run LangGraph Xiaohongshu copywriting workflow.')
    parser.add_argument('--topic', required=True, help='Topic for Xiaohongshu viral copywriting')
    args = parser.parse_args()

    service = XhsAgentService(DashScopeClient())
    result = await service.generate(args.topic)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(_main())