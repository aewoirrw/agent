from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Iterable, TYPE_CHECKING

from openai import OpenAI

from app.core.settings import settings

if TYPE_CHECKING:
    from app.services.vector_store import SearchResult


logger = logging.getLogger(__name__)


class ModelScopeReranker:
    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def enabled(self) -> bool:
        return bool(
            settings.rerank_enabled
            and settings.modelscope_reranker_api_key.strip()
            and settings.modelscope_reranker_base_url.strip()
            and settings.modelscope_reranker_model.strip()
        )

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=settings.modelscope_reranker_base_url.strip(),
                api_key=settings.modelscope_reranker_api_key.strip(),
            )
        return self._client

    async def rerank(self, query: str, candidates: Iterable['SearchResult'], top_k: int) -> list['SearchResult']:
        rows = list(candidates)
        if not self.enabled or not rows:
            return rows[:top_k]

        try:
            ranking = await asyncio.wait_for(
                asyncio.to_thread(self._rerank_sync, query, rows),
                timeout=max(1.0, float(settings.rerank_timeout_sec)),
            )
            if not ranking:
                return rows[:top_k]

            indexed = {index: item for index, item in enumerate(rows)}
            reordered: list['SearchResult'] = []
            seen: set[int] = set()
            for item in ranking:
                index = item.get('index')
                if not isinstance(index, int):
                    continue
                if index in indexed and index not in seen:
                    seen.add(index)
                    doc = indexed[index]
                    score = item.get('score')
                    if isinstance(score, (int, float)):
                        doc.score = float(score)
                    reordered.append(doc)

            for index, item in enumerate(rows):
                if index not in seen:
                    reordered.append(item)

            return reordered[:top_k]
        except Exception as e:
            logger.warning('ModelScope reranker failed, fallback to vector ranking: %s', repr(e))
            return rows[:top_k]

    def _rerank_sync(self, query: str, rows: list['SearchResult']) -> list[dict]:
        client = self._get_client()
        prompt = self._build_prompt(query, rows)
        response = client.chat.completions.create(
            model=settings.modelscope_reranker_model.strip(),
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a retrieval reranker. '
                        'Return only valid JSON with the shape '
                        '{"ranked":[{"index":0,"score":0.0}]}. '
                        'Higher score means more relevant. No markdown. No explanation.'
                    ),
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            stream=False,
        )
        text = ((response.choices or [None])[0].message.content or '').strip()
        return self._parse_ranked_json(text)

    @staticmethod
    def _build_prompt(query: str, rows: list['SearchResult']) -> str:
        candidates = []
        for index, row in enumerate(rows):
            candidates.append(f'[{index}] {row.content}')
        return (
            'Task: rerank the following candidate passages for the query by relevance.\n'
            f'Query:\n{query}\n\n'
            'Candidates:\n'
            + '\n\n'.join(candidates)
            + '\n\nReturn JSON only. Example: '
            + '{"ranked":[{"index":2,"score":0.98},{"index":0,"score":0.74}]}'
        )

    @staticmethod
    def _parse_ranked_json(text: str) -> list[dict]:
        if not text:
            return []

        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                return []
            try:
                obj = json.loads(match.group(0))
            except Exception:
                return []

        ranked = obj.get('ranked') if isinstance(obj, dict) else None
        if not isinstance(ranked, list):
            return []

        out: list[dict] = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            index = item.get('index')
            score = item.get('score', 0.0)
            if isinstance(index, int):
                try:
                    out.append({'index': index, 'score': float(score)})
                except Exception:
                    out.append({'index': index, 'score': 0.0})
        return out