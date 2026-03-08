from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
import json
import logging
import threading
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from app.core.settings import settings
from app.services.tools import AgentTools


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LocalEmbeddingStartupStatus:
    enabled: bool
    state: str = 'not_started'  # not_started|starting|ready|disabled|unavailable|failed
    progress: int = 0
    message: str = ''
    modelPath: str = ''
    requestedDevice: str = ''
    device: str = ''
    torchVersion: str | None = None
    cudaAvailable: bool | None = None
    gpuName: str | None = None
    deviceCount: int | None = None
    totalMemoryMB: int | None = None
    allocatedMemoryMB: int | None = None
    reservedMemoryMB: int | None = None
    startedAt: str | None = None
    finishedAt: str | None = None
    error: str | None = None


class DashScopeClient:
    def __init__(self) -> None:
        self._local_embedder = None
        self._local_embedder_lock = threading.Lock()
        self._local_embedding_warmed = False
        self._local_embedding_last_error: str | None = None
        self._local_embedding_status_lock = threading.Lock()
        self._local_embedding_status = LocalEmbeddingStartupStatus(
            enabled=bool(settings.local_embedding_enabled),
            state='disabled' if not settings.local_embedding_enabled else 'not_started',
            progress=100 if not settings.local_embedding_enabled else 0,
            message='local embedding disabled' if not settings.local_embedding_enabled else '',
            modelPath=str(settings.local_embedding_model_path),
            requestedDevice=str(getattr(settings, 'local_embedding_device', 'auto')),
            device='',
            torchVersion=None,
            cudaAvailable=None,
            gpuName=None,
            deviceCount=None,
            totalMemoryMB=None,
            allocatedMemoryMB=None,
            reservedMemoryMB=None,
            startedAt=None,
            finishedAt=_utc_now_iso() if not settings.local_embedding_enabled else None,
            error=None,
        )
        self._glm_client: OpenAI | None = None
        self.max_retries = 2
        self.retry_backoff_sec = 0.6

    def local_embedding_startup_status(self) -> dict:
        with self._local_embedding_status_lock:
            gpu_runtime = self._detect_cuda_runtime_info(self._local_embedding_status.device or None)
            self._merge_cuda_runtime_info_locked(gpu_runtime)
            # keep warmed flag reflected
            if self._local_embedding_warmed and self._local_embedding_status.state not in ('ready', 'disabled'):
                self._local_embedding_status.state = 'ready'
                self._local_embedding_status.progress = 100
                self._local_embedding_status.message = self._local_embedding_status.message or 'ready'
                self._local_embedding_status.finishedAt = self._local_embedding_status.finishedAt or _utc_now_iso()
            return asdict(self._local_embedding_status)

    def _set_local_embedding_status(
        self,
        *,
        state: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        requestedDevice: str | None = None,
        device: str | None = None,
        torchVersion: str | None = None,
        cudaAvailable: bool | None = None,
        gpuName: str | None = None,
        deviceCount: int | None = None,
        totalMemoryMB: int | None = None,
        allocatedMemoryMB: int | None = None,
        reservedMemoryMB: int | None = None,
        mark_started: bool = False,
        mark_finished: bool = False,
    ) -> None:
        with self._local_embedding_status_lock:
            st = self._local_embedding_status
            if state is not None:
                st.state = state
            if progress is not None:
                st.progress = max(0, min(100, int(progress)))
            if message is not None:
                st.message = message
            if error is not None:
                st.error = error
            st.modelPath = str(settings.local_embedding_model_path)
            st.enabled = bool(settings.local_embedding_enabled)
            st.requestedDevice = str(getattr(settings, 'local_embedding_device', 'auto')) if requestedDevice is None else str(requestedDevice)
            if device is not None:
                st.device = str(device)
            if torchVersion is not None:
                st.torchVersion = torchVersion
            if cudaAvailable is not None:
                st.cudaAvailable = bool(cudaAvailable)
            if gpuName is not None:
                st.gpuName = gpuName
            if deviceCount is not None:
                st.deviceCount = int(deviceCount)
            if totalMemoryMB is not None:
                st.totalMemoryMB = int(totalMemoryMB)
            if allocatedMemoryMB is not None:
                st.allocatedMemoryMB = int(allocatedMemoryMB)
            if reservedMemoryMB is not None:
                st.reservedMemoryMB = int(reservedMemoryMB)
            if mark_started and st.startedAt is None:
                st.startedAt = _utc_now_iso()
            if mark_finished:
                st.finishedAt = _utc_now_iso()

    def _merge_cuda_runtime_info_locked(self, info: dict[str, Any]) -> None:
        st = self._local_embedding_status
        if not info:
            return
        if info.get('gpuName') is not None:
            st.gpuName = info.get('gpuName')
        if info.get('deviceCount') is not None:
            st.deviceCount = int(info.get('deviceCount'))
        if info.get('totalMemoryMB') is not None:
            st.totalMemoryMB = int(info.get('totalMemoryMB'))
        if info.get('allocatedMemoryMB') is not None:
            st.allocatedMemoryMB = int(info.get('allocatedMemoryMB'))
        if info.get('reservedMemoryMB') is not None:
            st.reservedMemoryMB = int(info.get('reservedMemoryMB'))

    @staticmethod
    def _detect_torch_info() -> tuple[str | None, bool | None]:
        try:
            import torch
            torch_version = getattr(torch, '__version__', None)
            cuda_ok = None
            try:
                cuda_ok = bool(torch.cuda.is_available())
            except Exception:
                cuda_ok = None
            return torch_version, cuda_ok
        except Exception:
            return None, None

    @staticmethod
    def _detect_cuda_runtime_info(device: str | None = None) -> dict[str, Any]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
            if not cuda_available or device_count <= 0:
                return {
                    'deviceCount': device_count,
                }

            device_index = 0
            if device and isinstance(device, str) and device.startswith('cuda:'):
                try:
                    device_index = int(device.split(':', 1)[1])
                except Exception:
                    device_index = 0

            props = torch.cuda.get_device_properties(device_index)
            allocated = int(torch.cuda.memory_allocated(device_index) / (1024 * 1024))
            reserved = int(torch.cuda.memory_reserved(device_index) / (1024 * 1024))
            total = int(props.total_memory / (1024 * 1024))
            return {
                'gpuName': torch.cuda.get_device_name(device_index),
                'deviceCount': device_count,
                'totalMemoryMB': total,
                'allocatedMemoryMB': allocated,
                'reservedMemoryMB': reserved,
            }
        except Exception:
            return {}

    @staticmethod
    def _resolve_requested_device(requested: str, cuda_available: bool | None) -> tuple[str, str | None]:
        req = (requested or 'auto').strip().lower()
        if req in ('', 'auto'):
            if cuda_available is True:
                return 'cuda', None
            return 'cpu', None

        if req == 'cpu':
            return 'cpu', None

        if req.startswith('cuda'):
            if cuda_available is True:
                # allow cuda:0, cuda:1 etc.
                return requested.strip(), None
            return 'cpu', 'CUDA not available, fallback to CPU'

        # unknown value -> fallback
        return 'cpu', f'Unknown local_embedding_device={requested!r}, fallback to CPU'

    def _compute_local_embedding_device(self) -> tuple[str, str | None, str | None, bool | None]:
        requested = str(getattr(settings, 'local_embedding_device', 'auto') or 'auto')
        torch_version, cuda_available = self._detect_torch_info()
        resolved_device, note = self._resolve_requested_device(requested, cuda_available)
        return resolved_device, note, torch_version, cuda_available

    def validate_local_embedding_device_or_raise(self) -> tuple[str, str | None, str | None, bool | None]:
        requested = str(getattr(settings, 'local_embedding_device', 'cuda') or 'cuda').strip()
        resolved_device, note, torch_version, cuda_available = self._compute_local_embedding_device()
        requested_lower = requested.lower()
        gpu_runtime = self._detect_cuda_runtime_info(resolved_device)

        if requested_lower.startswith('cuda') and cuda_available is not True:
            err = (
                'GPU is required for local embedding but CUDA is not available '
                f'(requestedDevice={requested}, torchVersion={torch_version}, cudaAvailable={cuda_available})'
            )
            self._local_embedding_last_error = err
            self._set_local_embedding_status(
                state='failed',
                progress=100,
                message='GPU is required but not available',
                error=err,
                requestedDevice=requested,
                device='unavailable',
                torchVersion=torch_version,
                cudaAvailable=cuda_available,
                gpuName=gpu_runtime.get('gpuName'),
                deviceCount=gpu_runtime.get('deviceCount'),
                totalMemoryMB=gpu_runtime.get('totalMemoryMB'),
                allocatedMemoryMB=gpu_runtime.get('allocatedMemoryMB'),
                reservedMemoryMB=gpu_runtime.get('reservedMemoryMB'),
                mark_started=True,
                mark_finished=True,
            )
            raise RuntimeError(err)

        if requested_lower not in ('cpu', 'cuda', 'auto') and not requested_lower.startswith('cuda:'):
            err = f'Unsupported LOCAL_EMBEDDING_DEVICE value: {requested}'
            self._local_embedding_last_error = err
            self._set_local_embedding_status(
                state='failed',
                progress=100,
                message='Unsupported local embedding device setting',
                error=err,
                requestedDevice=requested,
                device='invalid',
                torchVersion=torch_version,
                cudaAvailable=cuda_available,
                gpuName=gpu_runtime.get('gpuName'),
                deviceCount=gpu_runtime.get('deviceCount'),
                totalMemoryMB=gpu_runtime.get('totalMemoryMB'),
                allocatedMemoryMB=gpu_runtime.get('allocatedMemoryMB'),
                reservedMemoryMB=gpu_runtime.get('reservedMemoryMB'),
                mark_started=True,
                mark_finished=True,
            )
            raise RuntimeError(err)

        return resolved_device, note, torch_version, cuda_available

    @property
    def enabled(self) -> bool:
        key = settings.llm_api_key
        return bool(key and key != 'your-api-key-here')

    def _get_client(self) -> OpenAI | None:
        if not self.enabled:
            return None
        if self._glm_client is None:
            kwargs = {'api_key': settings.llm_api_key}
            if (settings.llm_base_url or '').strip():
                kwargs['base_url'] = settings.llm_base_url.strip()
            self._glm_client = OpenAI(**kwargs)
        return self._glm_client

    async def chat(self, messages: list[dict], stream: bool = False):
        if not self.enabled:
            if stream:
                async def _mock_stream():
                    text = '当前未配置 LLM API Key，已进入本地降级回复模式。'
                    for i in range(0, len(text), 12):
                        yield text[i:i + 12]
                return _mock_stream()
            return '当前未配置 LLM API Key，已进入本地降级回复模式。'

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
            return '当前未配置 LLM API Key，已进入本地降级回复模式。', []

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
            self.validate_local_embedding_device_or_raise()
            local_vec = await asyncio.to_thread(self._local_embedding_from_model, text)
            if local_vec is not None:
                return local_vec
            raise RuntimeError(self._local_embedding_last_error or 'Local GPU embedding unavailable')

        if self.enabled:
            try:
                obj = await asyncio.to_thread(self._embedding_once_sync, text)
                emb = (((obj.get('data') or [{}])[0] or {}).get('embedding'))
                if emb:
                    return [float(v) for v in emb]
            except Exception as e:
                logger.warning('GLM embedding failed: %s', repr(e))
                raise

        raise RuntimeError('No embedding backend available')

    async def prewarm_local_embedding(self) -> bool:
        if not settings.local_embedding_enabled:
            self._set_local_embedding_status(
                state='disabled',
                progress=100,
                message='local embedding disabled',
                error=None,
                mark_finished=True,
            )
            return False

        if self._local_embedding_warmed:
            self._set_local_embedding_status(
                state='ready',
                progress=100,
                message='local embedding already warmed',
                error=None,
                mark_started=True,
                mark_finished=True,
            )
            return True

        resolved_device, device_note, torch_version, cuda_available = self.validate_local_embedding_device_or_raise()
        start_msg = f'starting local embedding model (device={resolved_device})'
        if device_note:
            start_msg = f'{start_msg} ({device_note})'
        self._set_local_embedding_status(
            state='starting',
            progress=3,
            message=start_msg,
            error=None,
            device=resolved_device,
            torchVersion=torch_version,
            cudaAvailable=cuda_available,
            mark_started=True,
        )
        logger.info('[startup] local embedding: starting (modelPath=%s, device=%s)', settings.local_embedding_model_path, resolved_device)

        ok = await asyncio.to_thread(
            self._prewarm_local_embedding_sync,
            resolved_device,
            device_note,
            torch_version,
            cuda_available,
        )
        return ok

    def _prewarm_local_embedding_sync(
        self,
        resolved_device: str,
        device_note: str | None,
        torch_version: str | None,
        cuda_available: bool | None,
    ) -> bool:
        try:
            self._set_local_embedding_status(progress=8, message='checking model directory')

            model_dir = Path(settings.local_embedding_model_path).expanduser()
            if not model_dir.is_absolute():
                model_dir = Path.cwd() / model_dir

            if not model_dir.exists():
                self._set_local_embedding_status(
                    state='unavailable',
                    progress=100,
                    message=f'model directory not found: {model_dir}',
                    error=None,
                    mark_finished=True,
                )
                logger.warning('[startup] local embedding: model dir not found: %s', str(model_dir))
                return False

            self._set_local_embedding_status(progress=20, message='importing sentence-transformers')
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
            except Exception as e:
                err = f'failed to import sentence-transformers: {repr(e)}'
                self._local_embedding_last_error = err
                self._set_local_embedding_status(
                    state='unavailable',
                    progress=100,
                    message='sentence-transformers not available',
                    error=err,
                    mark_finished=True,
                )
                logger.warning('[startup] local embedding: %s', err)
                return False

            # refresh device/torch info in case the environment changed between async and thread
            self._set_local_embedding_status(
                progress=35,
                message=(
                    f'preparing device={resolved_device}' + (f' ({device_note})' if device_note else '')
                ),
                device=resolved_device,
                torchVersion=torch_version,
                cudaAvailable=cuda_available,
            )

            self._set_local_embedding_status(progress=55, message=f'loading local embedding model (device={resolved_device})', device=resolved_device)
            embedder = self._get_local_embedder(model_dir, device=resolved_device)
            if embedder is None:
                err = self._local_embedding_last_error or 'failed to load local embedder'
                self._set_local_embedding_status(
                    state='failed',
                    progress=100,
                    message='failed to load local embedding model',
                    error=err,
                    device=resolved_device,
                    mark_finished=True,
                )
                logger.warning('[startup] local embedding: load failed: %s', err)
                return False

            self._set_local_embedding_status(progress=85, message=f'warming up embedding model (device={resolved_device})', device=resolved_device)
            _ = embedder.encode(['embedding model warmup'], normalize_embeddings=True)[0]
            self._local_embedding_warmed = True

            self._set_local_embedding_status(
                state='ready',
                progress=100,
                message=f'local embedding warmup completed (device={resolved_device})',
                error=None,
                device=resolved_device,
                mark_finished=True,
            )
            logger.info('[startup] local embedding: ready')
            return True
        except Exception as e:
            err = repr(e)
            self._local_embedding_last_error = err
            self._set_local_embedding_status(
                state='failed',
                progress=100,
                message='local embedding warmup failed',
                error=err,
                mark_finished=True,
            )
            logger.warning('[startup] local embedding: warmup failed: %s', err)
            return False

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
            raise RuntimeError('LLM client not configured')
        return client.chat.completions.create(**payload)

    def _embedding_once_sync(self, text: str):
        client = self._get_client()
        if client is None:
            raise RuntimeError('LLM client not configured')
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
            self._local_embedding_last_error = f'Local embedding model directory not found: {model_dir}'
            return None

        try:
            resolved_device, _, _, _ = self.validate_local_embedding_device_or_raise()
            embedder = self._get_local_embedder(model_dir, device=resolved_device)
            if embedder is None:
                return None

            vec = embedder.encode([text], normalize_embeddings=True)[0]
            return [float(v) for v in vec]
        except Exception as e:
            self._local_embedding_last_error = repr(e)
            return None

    def _get_local_embedder(self, model_dir: Path, device: str = 'cpu'):
        if self._local_embedder is not None:
            self._local_embedding_warmed = True
            return self._local_embedder

        with self._local_embedder_lock:
            if self._local_embedder is not None:
                self._local_embedding_warmed = True
                return self._local_embedder
            try:
                from sentence_transformers import SentenceTransformer
                self._local_embedder = SentenceTransformer(
                    str(model_dir),
                    device=device,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self._local_embedding_warmed = True
                self._local_embedding_last_error = None
                return self._local_embedder
            except Exception as e:
                self._local_embedding_last_error = repr(e)
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
            return 'LLM 鉴权失败（请检查 API Key）'
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
