from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    frac = idx - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    upload_ms = [float(r['upload_response_ms']) for r in rows]
    index_ms = [float(r['index_total_ms']) for r in rows if float(r['index_total_ms']) > 0]
    return {
        'count': float(len(rows)),
        'upload_p50_ms': percentile(upload_ms, 0.5),
        'upload_p95_ms': percentile(upload_ms, 0.95),
        'upload_avg_ms': statistics.mean(upload_ms) if upload_ms else 0.0,
        'index_p50_ms': percentile(index_ms, 0.5) if index_ms else 0.0,
        'index_p95_ms': percentile(index_ms, 0.95) if index_ms else 0.0,
        'index_avg_ms': statistics.mean(index_ms) if index_ms else 0.0,
    }


def print_summary(summary: dict[str, float]) -> None:
    print('\n== Upload Benchmark Summary ==')
    print(f"count={int(summary['count'])}")
    print(f"upload_p50_ms={summary['upload_p50_ms']:.2f}")
    print(f"upload_p95_ms={summary['upload_p95_ms']:.2f}")
    print(f"upload_avg_ms={summary['upload_avg_ms']:.2f}")
    print(f"index_p50_ms={summary['index_p50_ms']:.2f}")
    print(f"index_p95_ms={summary['index_p95_ms']:.2f}")
    print(f"index_avg_ms={summary['index_avg_ms']:.2f}")


async def run_once(client: httpx.AsyncClient, base_url: str, file_path: Path, label: str, iteration: int, poll_interval: float, poll_timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    with file_path.open('rb') as f:
        resp = await client.post(f'{base_url}/api/upload', files={'file': (file_path.name, f, 'text/markdown')})
    upload_response_ms = (time.perf_counter() - started) * 1000
    resp.raise_for_status()

    payload = resp.json()
    task = payload.get('data') or {}
    task_id = task.get('taskId')
    if not task_id:
        raise RuntimeError(f'missing taskId in response: {json.dumps(payload, ensure_ascii=False)}')

    poll_started = time.perf_counter()
    final_snap: dict[str, Any] | None = None
    deadline = time.perf_counter() + poll_timeout
    while time.perf_counter() < deadline:
        task_resp = await client.get(f'{base_url}/api/upload/tasks/{task_id}')
        task_resp.raise_for_status()
        final_snap = (task_resp.json().get('data') or {})
        if final_snap.get('status') in {'success', 'failed'}:
            break
        await asyncio_sleep(poll_interval)

    if not final_snap:
        raise RuntimeError('failed to fetch upload task snapshot')
    if final_snap.get('status') not in {'success', 'failed'}:
        raise RuntimeError(f'poll timeout for task {task_id}')

    row = {
        'label': label,
        'iteration': iteration,
        'file_name': file_path.name,
        'file_size': file_path.stat().st_size,
        'task_id': task_id,
        'upload_response_ms': round(upload_response_ms, 2),
        'index_total_ms': final_snap.get('durationMs') or 0,
        'status': final_snap.get('status') or 'unknown',
        'progress': final_snap.get('progress') or 0,
        'stage': final_snap.get('stage') or '',
        'total_chunks': final_snap.get('totalChunks') or 0,
        'completed_chunks': final_snap.get('completedChunks') or 0,
        'storage_mode': ((final_snap.get('extra') or {}).get('storageMode') or ''),
        'error': final_snap.get('error') or '',
        'poll_elapsed_ms': round((time.perf_counter() - poll_started) * 1000, 2),
    }
    return row


async def asyncio_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)


async def main_async(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip('/')
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.http_timeout) as client:
        for i in range(1, args.repeat + 1):
            row = await run_once(client, base_url, file_path, args.label, i, args.poll_interval, args.poll_timeout)
            rows.append(row)
            print(f"[{i}/{args.repeat}] upload_response_ms={row['upload_response_ms']} index_total_ms={row['index_total_ms']} status={row['status']} chunks={row['total_chunks']}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'label', 'iteration', 'file_name', 'file_size', 'task_id',
        'upload_response_ms', 'index_total_ms', 'poll_elapsed_ms',
        'status', 'progress', 'stage', 'total_chunks', 'completed_chunks',
        'storage_mode', 'error',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nCSV written to: {out_path}')
    print_summary(summarize(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Benchmark Python upload response and background indexing time')
    parser.add_argument('--base-url', default='http://127.0.0.1:9910')
    parser.add_argument('--file', default='uploads/upload_test.md')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--label', default='after')
    parser.add_argument('--output', default='benchmark_outputs/upload_benchmark_after.csv')
    parser.add_argument('--poll-interval', type=float, default=0.5)
    parser.add_argument('--poll-timeout', type=float, default=180.0)
    parser.add_argument('--http-timeout', type=float, default=180.0)
    return parser


if __name__ == '__main__':
    import asyncio

    args = build_parser().parse_args()
    asyncio.run(main_async(args))
