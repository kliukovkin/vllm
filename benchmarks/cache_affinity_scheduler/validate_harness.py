# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only dry-run validator for the cache_affinity_scheduler harness.

Runs without loading a real model by injecting a lightweight mock engine.
Validates:

1. ``BenchmarkConfig`` and ``RequestRecord`` dataclasses are importable and
   round-trip through JSON.
2. ``replay_trace()`` calls the engine's ``generate()`` method and returns
   one latency per request.
3. ``aggregate()`` computes sane statistics (mean > 0, p95 >= p50 >= 0).
4. ``save_trace_to_jsonl()`` / ``load_trace_from_jsonl()`` round-trip data.
5. ``write_result()`` creates a valid JSON file on disk.
6. ``compare_results()`` returns the expected keys.

Usage::

    python -m benchmarks.cache_affinity_scheduler.validate_harness
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from benchmarks.cache_affinity_scheduler.harness import (
    BenchmarkConfig,
    BenchmarkResult,
    RequestRecord,
    aggregate,
    compare_results,
    load_trace_from_jsonl,
    replay_trace,
    save_trace_to_jsonl,
    write_result,
)
from vllm import SamplingParams


class _MockEngine:
    """Minimal engine stub — returns empty RequestOutput-like objects."""

    def __init__(self, delay_s: float = 0.001) -> None:
        self._delay_s = delay_s
        self.call_count = 0

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        **kwargs: Any,
    ) -> list[MagicMock]:
        time.sleep(self._delay_s * len(prompts))
        self.call_count += 1
        return [MagicMock() for _ in prompts]


def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK  : {msg}")


def main() -> None:
    print("=== validate_harness: CPU dry-run ===")

    # ------------------------------------------------------------------
    # 1. Dataclass round-trip
    # ------------------------------------------------------------------
    print("\n[1] Dataclass instantiation and JSON round-trip")
    cfg = BenchmarkConfig(model="mock-model", max_num_seqs=8)
    _check(cfg.model == "mock-model", "BenchmarkConfig.model")
    _check(cfg.max_num_seqs == 8, "BenchmarkConfig.max_num_seqs")

    records = [
        RequestRecord(prompt="hello world", output_len=16, group_id="g0"),
        RequestRecord(prompt="foo bar baz", output_len=8, group_id="g0"),
        RequestRecord(prompt="unique prompt", output_len=32, group_id="g1"),
    ]
    _check(len(records) == 3, "records list length")
    _check(records[0].group_id == "g0", "RequestRecord.group_id")

    # ------------------------------------------------------------------
    # 2. replay_trace()
    # ------------------------------------------------------------------
    print("\n[2] replay_trace() with mock engine")
    engine = _MockEngine(delay_s=0.005)
    latencies = replay_trace(records, engine, max_new_tokens=16)
    _check(
        len(latencies) == len(records),
        f"one latency per request ({len(latencies)} vs {len(records)})",
    )
    _check(all(lat >= 0 for lat in latencies), "all latencies >= 0")
    _check(engine.call_count == 1, f"single generate() call (got {engine.call_count})")

    # batch_size=1 → one call per request
    engine2 = _MockEngine(delay_s=0.001)
    latencies2 = replay_trace(records, engine2, max_new_tokens=16, batch_size=1)
    _check(len(latencies2) == len(records), "latencies with batch_size=1")
    _check(
        engine2.call_count == len(records),
        f"one call per request with batch_size=1 ({engine2.call_count})",
    )

    # ------------------------------------------------------------------
    # 3. aggregate()
    # ------------------------------------------------------------------
    print("\n[3] aggregate() statistics")
    result = aggregate(latencies, "test_scenario", "mock_scheduler")
    _check(isinstance(result, BenchmarkResult), "aggregate() returns BenchmarkResult")
    _check(result.mean_latency_s > 0, "mean_latency_s > 0")
    _check(result.p95_latency_s >= result.p50_latency_s, "p95 >= p50")
    _check(result.p50_latency_s >= 0, "p50 >= 0")
    _check(result.throughput_req_s > 0, "throughput_req_s > 0")
    _check(result.num_requests == len(records), f"num_requests == {len(records)}")

    # ------------------------------------------------------------------
    # 4. Trace I/O
    # ------------------------------------------------------------------
    print("\n[4] save_trace_to_jsonl / load_trace_from_jsonl")
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl"
        save_trace_to_jsonl(records, trace_path)
        _check(trace_path.exists(), "trace file created")
        loaded = load_trace_from_jsonl(trace_path)
        _check(
            len(loaded) == len(records),
            f"loaded {len(loaded)} records (expected {len(records)})",
        )
        _check(loaded[0].prompt == records[0].prompt, "prompt round-trips")
        _check(loaded[0].output_len == records[0].output_len, "output_len round-trips")
        _check(loaded[0].group_id == records[0].group_id, "group_id round-trips")

    # ------------------------------------------------------------------
    # 5. write_result()
    # ------------------------------------------------------------------
    print("\n[5] write_result()")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "result.json"
        write_result(result, out_path)
        _check(out_path.exists(), "result file created")
        with open(out_path) as f:
            d = json.load(f)
        _check(d["scenario"] == "test_scenario", "scenario round-trips")
        _check(d["num_requests"] == len(records), "num_requests round-trips")

    # ------------------------------------------------------------------
    # 6. compare_results()
    # ------------------------------------------------------------------
    print("\n[6] compare_results()")
    result_b = aggregate([0.1, 0.12, 0.11], "baseline", "sched_b")
    result_a = aggregate([0.08, 0.09, 0.085], "affinity", "sched_a")
    cmp = compare_results(result_b, result_a)
    _check("latency_improvement_pct" in cmp, "latency_improvement_pct key")
    _check("throughput_improvement_pct" in cmp, "throughput_improvement_pct key")
    _check(
        cmp["latency_improvement_pct"] > 0,
        f"affinity latency improvement > 0 ({cmp['latency_improvement_pct']:.2f}%)",
    )

    # ------------------------------------------------------------------
    # 7. Edge cases
    # ------------------------------------------------------------------
    print("\n[7] Edge cases")
    empty_lats = replay_trace([], engine, max_new_tokens=16)
    _check(empty_lats == [], "empty trace returns []")

    try:
        aggregate([], "empty", "sched")
        _check(False, "aggregate([]) should raise ValueError")
    except ValueError:
        _check(True, "aggregate([]) raises ValueError")

    print("\n=== validate_harness: ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
    sys.exit(0)
