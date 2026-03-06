from __future__ import annotations

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
    def __init__(self, dashscope) -> None:
        self.dashscope = dashscope
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

    async def index_file(self, file_path: str) -> None:
        text = Path(file_path).read_text(encoding='utf-8')
        chunks = self.chunker.chunk_document(text)
        if not chunks:
            return

        docs = []
        local_docs = []
        for chunk in chunks:
            emb = await self.dashscope.embedding(chunk.content)
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

        if self._milvus is not None:
            try:
                self._milvus.insert(collection_name=settings.milvus_collection, data=docs)
                return
            except Exception:
                pass

        source = str(file_path)
        self._in_memory = [x for x in self._in_memory if json.loads(x['metadata']).get('_source') != source]
        self._in_memory.extend(local_docs)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        qv = await self.dashscope.embedding(query)

        if self._milvus is not None:
            try:
                res = self._milvus.search(
                    collection_name=settings.milvus_collection,
                    data=[qv],
                    limit=top_k,
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
                return out
            except Exception:
                pass

        scored = []
        for doc in self._in_memory:
            score = self._cosine(qv, doc['vector'])
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(id=item['id'], content=item['content'], score=float(score), metadata=item['metadata'])
            for score, item in scored[:top_k]
        ]

    def health(self) -> dict:
        if self._milvus is not None:
            try:
                cols = self._milvus.list_collections()
                return {'ok': True, 'collections': cols}
            except Exception as e:
                return {'ok': False, 'error': str(e)}

        return {'ok': True, 'collections': ['in_memory_docs'], 'mode': 'memory'}

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
