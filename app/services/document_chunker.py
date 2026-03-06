from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from app.core.settings import settings


@dataclass
class DocumentChunk:
    content: str
    chunk_index: int
    title: Optional[str] = None


class DocumentChunker:
    def chunk_document(self, content: str) -> list[DocumentChunk]:
        if not content.strip():
            return []

        max_size = settings.document_chunk_max_size
        overlap = settings.document_chunk_overlap
        chunks: list[DocumentChunk] = []

        start = 0
        idx = 0
        while start < len(content):
            end = min(start + max_size, len(content))
            part = content[start:end].strip()
            if part:
                chunks.append(DocumentChunk(content=part, chunk_index=idx))
                idx += 1
            if end == len(content):
                break
            start = max(0, end - overlap)

        return chunks
