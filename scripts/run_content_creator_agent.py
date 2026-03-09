from __future__ import annotations

import argparse
import asyncio
import json

from app.services.content_creator_service import ContentCreatorService
from app.services.dashscope_client import DashScopeClient
from app.services.modelscope_reranker import ModelScopeReranker
from app.services.tools import AgentTools
from app.services.vector_store import VectorStore


async def _main() -> None:
    parser = argparse.ArgumentParser(description='Run LangGraph content creator agent.')
    parser.add_argument('--goal', required=True, help='Content creation goal or niche description')
    parser.add_argument('--platform', default='xiaohongshu', help='Target platform')
    parser.add_argument('--audience', default='general', help='Target audience')
    parser.add_argument('--seed-topic', default='', help='Optional initial topic')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum generation-review loops')
    args = parser.parse_args()

    dashscope = DashScopeClient()
    vector_store = VectorStore(dashscope, ModelScopeReranker())
    tools = AgentTools(vector_store)
    service = ContentCreatorService(dashscope, tools)

    await tools.ensure_runtime_tools()
    try:
        result = await service.run(
            goal=args.goal,
            platform=args.platform,
            audience=args.audience,
            seed_topic=args.seed_topic,
            max_iterations=args.max_iterations,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await tools.close_runtime_tools()


if __name__ == '__main__':
    asyncio.run(_main())