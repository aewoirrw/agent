from __future__ import annotations

import asyncio
import hashlib
import math
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from zai import ZhipuAiClient

from app.core.settings import settings
from app.services.tools import AgentTools


logger = logging.getLogger(__name__)


class DashScopeClient:
    def __init__(self) -> None:
        self._local_embedder = None
        self._glm_client: ZhipuAiClient | None = None
        self.max_retries = 2
        self.retry_backoff_sec = 0.6

    @property
    def enabled(self) -> bool:
        key = settings.llm_api_key
        return bool(key and key != 'your-api-key-here')

    def _get_client(self) -> ZhipuAiClient | None:
        if not self.enabled:
            return None
        if self._glm_client is None:
            self._glm_client = ZhipuAiClient(api_key=settings.llm_api_key)
        return self._glm_client

    async def chat(self, messages: list[dict], stream: bool = False):
        if not self.enabled:
            if stream:
                async def _mock_stream():
                    text = '当前未配置 GLM API Key（ZHIPU_API_KEY），已进入本地降级回复模式。'
                    for i in range(0, len(text), 12):
                        yield text[i:i + 12]
                return _mock_stream()
            return '当前未配置 GLM API Key（ZHIPU_API_KEY），已进入本地降级回复模式。'

        payload = {
            'model': settings.chat_model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 4096,
        }

        try:
            if stream:
                async def _stream_gen():
                    text = await self._chat_text_with_retries(payload)
                    for i in range(0, len(text), 12):
                        yield text[i:i + 12]
                return _stream_gen()

            return await self._chat_text_with_retries(payload)
        except Exception as e:
            logger.warning('GLM chat unexpected error: %s', repr(e))
            return self._fallback_message(e)

    async def chat_with_tools(self, messages: list[dict], tools: AgentTools, max_rounds: int = 4) -> str:
        answer, _ = await self.chat_with_tools_trace(messages, tools, max_rounds=max_rounds)
        return answer

    async def chat_with_tools_trace(
        self,
        messages: list[dict],
        tools: AgentTools,
        max_rounds: int = 4,
    ) -> tuple[str, list[dict]]:
        if not self.enabled:
            return '当前未配置 GLM API Key（ZHIPU_API_KEY），已进入本地降级回复模式。', []

        tool_trace: list[dict] = []
        last_err = None
        for attempt in range(self.max_retries + 1):
            convo = list(messages)
            try:
                for _ in range(max_rounds):
                    payload = {
                        'model': settings.chat_model,
                        'messages': convo,
                        'temperature': 0.7,
                        'max_tokens': 4096,
                        'tools': tools.definitions,
                        'tool_choice': 'auto',
                    }
                    obj = await self._chat_object_once(payload)
                    msg = ((obj.get('choices') or [{}])[0] or {}).get('message') or {}

                    tool_calls = msg.get('tool_calls') or []
                    content = msg.get('content') or ''

                    convo.append(
                        {
                            'role': 'assistant',
                            'content': content,
                            **({'tool_calls': tool_calls} if tool_calls else {}),
                        }
                    )

                    if not tool_calls:
                        return content, tool_trace

                    for tc in tool_calls:
                        func = tc.get('function', {})
                        name = func.get('name')
                        args_raw = func.get('arguments') or '{}'
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            args = {}

                        tool_trace.append(
                            {
                                'event': 'tool_started',
                                'tool': name,
                                'arguments': args,
                                'resultPreview': '',
                            }
                        )

                        tool_result = await tools.run(name, args)
                        tool_trace.append(
                            {
                                'event': 'tool_finished',
                                'tool': name,
                                'arguments': args,
                                'resultPreview': (tool_result or '')[:300],
                            }
                        )
                        convo.append(
                            {
                                'role': 'tool',
                                'tool_call_id': tc.get('id'),
                                'name': name,
                                'content': tool_result,
                            }
                        )

                return '工具调用轮次达到上限，请简化问题后重试。', tool_trace
            except Exception as e:
                last_err = e
                logger.warning('GLM tool-chat failed (attempt %s/%s): %s', attempt + 1, self.max_retries + 1, repr(e))
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * (attempt + 1))

        return self._fallback_message(last_err), tool_trace

    async def embedding(self, text: str) -> list[float]:
        if settings.local_embedding_enabled:
            local_vec = self._local_embedding_from_model(text)
            if local_vec is not None:
                return local_vec

        if self.enabled:
            try:
                obj = await asyncio.to_thread(self._embedding_once_sync, text)
                emb = (((obj.get('data') or [{}])[0] or {}).get('embedding'))
                if emb:
                    return [float(v) for v in emb]
            except Exception as e:
                logger.warning('GLM embedding failed, use local embedding: %s', repr(e))
                return self._local_embedding(text)

        return self._local_embedding(text)

    async def _chat_text_with_retries(self, payload: dict) -> str:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                obj = await self._chat_object_once(payload)
                msg = ((obj.get('choices') or [{}])[0] or {}).get('message') or {}
                return msg.get('content') or ''
            except Exception as e:
                last_err = e
                logger.warning('GLM chat failed (attempt %s/%s): %s', attempt + 1, self.max_retries + 1, repr(e))
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * (attempt + 1))
        return self._fallback_message(last_err)

    async def _chat_object_once(self, payload: dict) -> dict:
        response = await asyncio.to_thread(self._chat_once_sync, payload)
        return self._to_dict(response)

    def _chat_once_sync(self, payload: dict):
        client = self._get_client()
        if client is None:
            raise RuntimeError('GLM client not configured')
        return client.chat.completions.create(**payload)

    def _embedding_once_sync(self, text: str):
        client = self._get_client()
        if client is None:
            raise RuntimeError('GLM client not configured')
        return client.embeddings.create(model=settings.embedding_model, input=text)

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            try:
                return obj.dict()
            except Exception:
                return {}
        return {}

    def _local_embedding_from_model(self, text: str):
        model_dir = Path(settings.local_embedding_model_path).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path.cwd() / model_dir
        if not model_dir.exists():
            return None

        try:
            embedder = self._get_local_embedder(model_dir)
            if embedder is None:
                return None

            vec = embedder.encode([text], normalize_embeddings=True)[0]
            return [float(v) for v in vec]
        except Exception:
            return None

    def _get_local_embedder(self, model_dir: Path):
        if self._local_embedder is not None:
            return self._local_embedder

        try:
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(str(model_dir), device='cpu', trust_remote_code=True)
            return self._local_embedder
        except Exception:
            return None

    def _local_embedding(self, text: str, dim: int = 256) -> list[float]:
        vals = [0.0] * dim
        for tok in text.split():
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
            vals[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def _fallback_message(self, err: Exception | None) -> str:
        return f'{self._classify_error(err)}，已切换为本地降级回复。'

    @staticmethod
    def _classify_error(err: Exception | None) -> str:
        if err is None:
            return '调用 GLM 失败'

        text = str(err).lower()
        if '401' in text or '403' in text or 'unauthorized' in text or 'forbidden' in text:
            return 'GLM 鉴权失败（请检查 ZHIPU_API_KEY）'
        if '429' in text or 'rate limit' in text:
            return 'GLM 调用被限流（429）'
        if '400' in text or 'bad request' in text:
            return 'GLM 请求参数错误（请检查模型名或入参）'
        if '500' in text or '502' in text or '503' in text or '504' in text:
            return 'GLM 服务异常（5xx）'
        if 'name resolution' in text or 'getaddrinfo' in text:
            return 'GLM DNS 解析失败'
        if 'timed out' in text or 'timeout' in text:
            return 'GLM 请求超时'

        return '调用 GLM 失败'


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if len(av) != len(bv) or not av:
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(y * y for y in bv)) or 1.0
    return dot / (na * nb)
