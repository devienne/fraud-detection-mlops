"""Load test for the fraud API: latency percentiles (p50 / p95 / p99) and throughput.

Replays real transactions from the scoring file in three modes:
  1. single requests, one client       -> pure per-request latency
  2. single requests, concurrent clients -> latency under load + throughput
  3. batches of 100, one client        -> per-transaction cost of batching

Usage:
  python scripts/load_test.py --url http://127.0.0.1:8001 --requests 1000 --concurrency 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["datahora_transacao", "nome_estabelecimento", "tipo_estabelecimento", "valor_transacao",
          "cidade", "device_type", "is_new_device"]


def load_transactions(path: Path, n: int, seed: int = 7) -> list[dict]:
    df = pd.read_csv(path).sample(n, replace=n > 0, random_state=seed)
    df["datahora_transacao"] = df["datahora_transacao"].str.replace(" ", "T")
    records = df[FIELDS].to_dict(orient="records")
    for i, r in enumerate(records):
        r["transaction_id"] = f"load-{i}"
        r["valor_transacao"] = float(r["valor_transacao"])
        r["is_new_device"] = bool(r["is_new_device"])
    return records


def timed_post(client: httpx.Client, path: str, body: dict) -> tuple[float, int]:
    start = time.perf_counter()
    status = client.post(path, json=body).status_code
    return (time.perf_counter() - start) * 1000, status


def run(url: str, path: str, bodies: list[dict], concurrency: int) -> dict:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(base_url=url, timeout=30, limits=limits) as client:
        timed_post(client, path, bodies[0])          # warm-up (connection + caches), not measured
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda b: timed_post(client, path, b), bodies))
        wall = time.perf_counter() - start
    latencies = np.array([r[0] for r in results])
    errors = sum(1 for r in results if r[1] != 200)
    return {"requests": len(bodies), "errors": errors,
            "p50": np.percentile(latencies, 50), "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99), "max": latencies.max(),
            "throughput_rps": len(bodies) / wall}


def report(name: str, r: dict, per_request: int = 1) -> None:
    tps = r["throughput_rps"] * per_request
    print(f"{name:38s} p50 {r['p50']:7.1f} ms | p95 {r['p95']:7.1f} ms | p99 {r['p99']:7.1f} ms | "
          f"max {r['max']:7.1f} ms | {r['throughput_rps']:6.1f} req/s = {tps:7.0f} tx/s | errors {r['errors']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "transacoes_jan2026.csv"))
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    txs = load_transactions(Path(args.data), args.requests)
    print(f"Target {args.url}  |  {args.requests} real January transactions\n")

    report("1. single, 1 client", run(args.url, "/predict", txs[: args.requests // 2], 1))
    report(f"2. single, {args.concurrency} concurrent clients", run(args.url, "/predict", txs, args.concurrency))
    batches = [{"transactions": txs[i:i + args.batch_size]} for i in range(0, len(txs), args.batch_size)]
    report(f"3. batches of {args.batch_size}, 1 client", run(args.url, "/predict/batch", batches, 1), args.batch_size)


if __name__ == "__main__":
    main()
