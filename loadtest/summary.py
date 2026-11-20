from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--p95-slo-ms", type=float, default=500.0)
    parser.add_argument("--error-slo-percent", type=float, default=1.0)
    args = parser.parse_args()

    print("| Users | Requests | Req/s | p50 ms | p95 ms | p99 ms | Errors % | SLO |")
    print("|---:|---:|---:|---:|---:|---:|---:|:---:|")
    sorted_results = sorted(args.results, key=lambda item: int(Path(item).stem.removeprefix("w")))
    for result in sorted_results:
        path = Path(result)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        latencies = [row["latency_ms"] for row in rows]
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        duration = (
            json.loads(metadata_path.read_text(encoding="utf-8"))["elapsed_seconds"]
            if metadata_path.exists()
            else 0
        )
        throughputs = len(rows) / duration if duration else 0
        errors = sum(row["status"] != 200 for row in rows)
        error_percent = 100 * errors / len(rows) if rows else 100
        p95 = percentile(latencies, 0.95)
        ok = p95 <= args.p95_slo_ms and error_percent <= args.error_slo_percent
        users = path.stem.replace("w", "")
        print(
            f"| {users} | {len(rows)} | {throughputs:.2f} | "
            f"{percentile(latencies, 0.50):.1f} | {p95:.1f} | "
            f"{percentile(latencies, 0.99):.1f} | {error_percent:.2f} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )


if __name__ == "__main__":
    main()
