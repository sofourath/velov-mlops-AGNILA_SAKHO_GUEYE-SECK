from __future__ import annotations

import argparse
import concurrent.futures
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


PAYLOAD = {
    "station_id": 3,
    "timestamp": "2026-11-20T10:30:00+01:00",
    "capacity": 20,
    "bikes_available": 12,
    "temperature": 14.5,
    "is_raining": False,
}


def worker(url: str, ssl_context: ssl.SSLContext, duration: float) -> list[dict]:
    rows: list[dict] = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        body = json.dumps(PAYLOAD).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, context=ssl_context, timeout=10) as response:
                response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            exc.read()
            status = exc.code
        except Exception as exc:
            status = 0
            error = type(exc).__name__
        else:
            error = ""
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({"latency_ms": elapsed, "status": status, "error": error if "error" in locals() else ""})
        if "error" in locals():
            error = ""
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--cacert", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ssl_context = ssl.create_default_context(cafile=args.cacert)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as executor:
        futures = [executor.submit(worker, args.url, ssl_context, args.duration) for _ in range(args.users)]
        rows = [row for future in futures for row in future.result()]
    elapsed = time.perf_counter() - started

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "users": args.users}),
        encoding="utf-8",
    )
    print(f"{len(rows)} requetes, {len(rows) / elapsed:.2f} req/s, resultats: {output}")


if __name__ == "__main__":
    main()
