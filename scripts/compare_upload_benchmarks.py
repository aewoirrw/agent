from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def metric(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = [float(r[key]) for r in rows if r.get(key) not in (None, '', '0')]
    if not values:
        return {'avg': 0.0, 'p50': 0.0, 'p95': 0.0}
    values = sorted(values)

    def pct(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    return {'avg': statistics.mean(values), 'p50': pct(0.5), 'p95': pct(0.95)}


def improve(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return ((before - after) / before) * 100.0


def print_compare(title: str, before_stats: dict[str, float], after_stats: dict[str, float]) -> None:
    print(f'\n== {title} ==')
    for key in ('avg', 'p50', 'p95'):
        b = before_stats[key]
        a = after_stats[key]
        print(f'{key}: before={b:.2f}ms after={a:.2f}ms improvement={improve(b, a):.2f}%')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two upload benchmark CSV files')
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    args = parser.parse_args()

    before_rows = read_csv(Path(args.before).resolve())
    after_rows = read_csv(Path(args.after).resolve())

    print_compare('Upload Response Time', metric(before_rows, 'upload_response_ms'), metric(after_rows, 'upload_response_ms'))
    print_compare('Background Index Time', metric(before_rows, 'index_total_ms'), metric(after_rows, 'index_total_ms'))


if __name__ == '__main__':
    main()
