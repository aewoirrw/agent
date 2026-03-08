from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UploadTask:
    task_id: str
    file_name: str
    file_path: str
    file_size: int
    status: str = 'queued'
    progress: int = 0
    stage: str = 'queued'
    message: str = '等待开始'
    total_chunks: int = 0
    completed_chunks: int = 0
    started_at: int = field(default_factory=lambda: int(time.time() * 1000))
    finished_at: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'taskId': self.task_id,
            'fileName': self.file_name,
            'filePath': self.file_path,
            'fileSize': self.file_size,
            'status': self.status,
            'progress': self.progress,
            'stage': self.stage,
            'message': self.message,
            'totalChunks': self.total_chunks,
            'completedChunks': self.completed_chunks,
            'startedAt': self.started_at,
            'finishedAt': self.finished_at,
            'durationMs': self.duration_ms,
            'error': self.error,
            'extra': self.extra,
        }


class UploadTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, UploadTask] = {}
        self._lock = threading.Lock()

    def create_task(self, file_name: str, file_path: str, file_size: int) -> UploadTask:
        task = UploadTask(
            task_id=f'upload_{uuid.uuid4().hex}',
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Optional[UploadTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = max(0, min(int(progress), 100))
            if stage is not None:
                task.stage = stage
            if message is not None:
                task.message = message
            if total_chunks is not None:
                task.total_chunks = int(total_chunks)
            if completed_chunks is not None:
                task.completed_chunks = int(completed_chunks)
            if error is not None:
                task.error = error
            if extra:
                task.extra.update(extra)

    def mark_running(self, task_id: str) -> None:
        self.update(task_id, status='running', progress=1, stage='starting', message='后台索引任务已启动')

    def mark_success(self, task_id: str, *, extra: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = 'success'
            task.progress = 100
            task.stage = 'completed'
            task.message = '索引完成'
            task.finished_at = int(time.time() * 1000)
            task.duration_ms = task.finished_at - task.started_at
            if extra:
                task.extra.update(extra)

    def mark_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = 'failed'
            task.stage = 'failed'
            task.message = '索引失败'
            task.error = error
            task.finished_at = int(time.time() * 1000)
            task.duration_ms = task.finished_at - task.started_at

    def snapshot(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else task.to_dict()
