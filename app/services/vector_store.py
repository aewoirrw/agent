from __future__ import annotations

import asyncio
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.services.document_chunker import DocumentChunker

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None


@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    metadata: str


class VectorStore:
    def __init__(self, dashscope, reranker=None) -> None:
        self.dashscope = dashscope
        self.reranker = reranker
        self.chunker = DocumentChunker()
        self._in_memory: list[dict[str, Any]] = []
        self._milvus = None

        if MilvusClient is not None:
            try:
                uri = settings.milvus_uri.strip() or f'http://{settings.milvus_host}:{settings.milvus_port}'
                kwargs = {'uri': uri}
                if settings.milvus_token.strip():
                    kwargs['token'] = settings.milvus_token.strip()
                if settings.milvus_db_name.strip():
                    kwargs['db_name'] = settings.milvus_db_name.strip()

                self._milvus = MilvusClient(**kwargs)
                self._ensure_collection()
            except Exception:
                self._milvus = None

    def _ensure_collection(self) -> None:
        if self._milvus is None:
            return
        has = self._milvus.has_collection(settings.milvus_collection)
        if has:
            return
        dim = 1024 if self.dashscope.enabled else 256
        self._milvus.create_collection(
            collection_name=settings.milvus_collection,
            dimension=dim,
            metric_type='COSINE',
            consistency_level='Strong'
        )

    async def index_file(self, file_path: str, progress_callback=None) -> dict[str, Any]:
        await self._notify_progress(progress_callback, stage='reading_file', progress=5, message='正在读取文件')
        text = await asyncio.to_thread(Path(file_path).read_text, encoding='utf-8')

        await self._notify_progress(progress_callback, stage='chunking', progress=12, message='正在进行文档分片')
        chunks = await asyncio.to_thread(self.chunker.chunk_document, text)
        if not chunks:
            await self._notify_progress(progress_callback, stage='completed', progress=100, message='文件为空，无需索引')
            return {'totalChunks': 0, 'storageMode': 'none'}

        total_chunks = len(chunks)
        await self._notify_progress(
            progress_callback,
            stage='embedding',
            progress=18,
            message=f'开始生成 embedding（共 {total_chunks} 个分片）',
            total_chunks=total_chunks,
            completed_chunks=0,
        )

        concurrency = max(1, int(settings.embedding_concurrency))
        semaphore = asyncio.Semaphore(concurrency)

        async def embed_one(chunk):
            async with semaphore:
                emb = await self.dashscope.embedding(chunk.content)
                return chunk, emb

        tasks = [embed_one(chunk) for chunk in chunks]
        embedded_results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            chunk, emb = await coro
            embedded_results.append((chunk, emb))
            completed += 1
            progress = 18 + int((completed / max(total_chunks, 1)) * 62)
            await self._notify_progress(
                progress_callback,
                stage='embedding',
                progress=progress,
                message=f'正在生成 embedding（{completed}/{total_chunks}）',
                total_chunks=total_chunks,
                completed_chunks=completed,
            )

        embedded_results.sort(key=lambda item: item[0].chunk_index)

        docs = []
        local_docs = []
        for chunk, emb in embedded_results:
            metadata_json = json.dumps({'_source': file_path, 'chunkIndex': chunk.chunk_index}, ensure_ascii=False)

            local_docs.append({
                'id': f"{Path(file_path).name}_{chunk.chunk_index}",
                'vector': emb,
                'content': chunk.content,
                'metadata': metadata_json,
            })

            docs.append({
                'vector': emb,
                'content': chunk.content,
                'metadata': metadata_json,
            })

        await self._notify_progress(progress_callback, stage='writing_vectors', progress=90, message='正在写入向量库', total_chunks=total_chunks, completed_chunks=total_chunks)
        if self._milvus is not None:
            try:
                self._milvus.insert(collection_name=settings.milvus_collection, data=docs)
                await self._notify_progress(progress_callback, stage='completed', progress=100, message='Milvus 写入完成', total_chunks=total_chunks, completed_chunks=total_chunks)
                return {'totalChunks': total_chunks, 'storageMode': 'milvus'}
            except Exception:
                pass

        await self._notify_progress(progress_callback, stage='fallback_memory', progress=95, message='Milvus 不可用，回退到内存索引', total_chunks=total_chunks, completed_chunks=total_chunks)
        source = str(file_path)
        self._in_memory = [x for x in self._in_memory if json.loads(x['metadata']).get('_source') != source]
        self._in_memory.extend(local_docs)
        await self._notify_progress(progress_callback, stage='completed', progress=100, message='内存索引完成', total_chunks=total_chunks, completed_chunks=total_chunks)
        return {'totalChunks': total_chunks, 'storageMode': 'memory'}

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        candidate_limit = max(
            int(top_k),
            int(settings.rerank_candidate_k) if getattr(self.reranker, 'enabled', False) else int(top_k),
        )
        qv = await self.dashscope.embedding(query)

        if self._milvus is not None:
            try:
                res = self._milvus.search(
                    collection_name=settings.milvus_collection,
                    data=[qv],
                    limit=candidate_limit,
                    output_fields=['content', 'metadata']
                )
                out: list[SearchResult] = []
                for hit in res[0]:
                    entity = hit.get('entity', {})
                    out.append(SearchResult(
                        id=str(hit.get('id')),
                        score=float(hit.get('distance', 0.0)),
                        content=entity.get('content', ''),
                        metadata=entity.get('metadata', '{}')
                    ))
                if getattr(self.reranker, 'enabled', False):
                    return await self.reranker.rerank(query, out, top_k)
                return out[:top_k]
            except Exception:
                pass

        scored = []
        for doc in self._in_memory:
            score = self._cosine(qv, doc['vector'])
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        rows = [
            SearchResult(id=item['id'], content=item['content'], score=float(score), metadata=item['metadata'])
            for score, item in scored[:candidate_limit]
        ]
        if getattr(self.reranker, 'enabled', False):
            return await self.reranker.rerank(query, rows, top_k)
        return rows[:top_k]

    def health(self) -> dict:
        if self._milvus is not None:
            try:
                cols = self._milvus.list_collections()
                return {'ok': True, 'collections': cols}
            except Exception as e:
                return {'ok': False, 'error': str(e)}

        return {'ok': True, 'collections': ['in_memory_docs'], 'mode': 'memory'}

    @staticmethod
    async def _notify_progress(callback, **payload) -> None:
        if callback is None:
            return
        result = callback(payload)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _cosine(a, b) -> float:
        av = list(a)
        bv = list(b)
        if len(av) != len(bv) or not av:
            return 0.0
        dot = sum(x * y for x, y in zip(av, bv))
        na = math.sqrt(sum(x * x for x in av)) or 1.0
        nb = math.sqrt(sum(y * y for y in bv)) or 1.0
        return dot / (na * nb)
