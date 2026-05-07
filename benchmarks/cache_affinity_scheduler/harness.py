# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness utilities for CacheAffinityScheduler benchmarks.

All four benchmark scripts (synthetic, RAG, ShareGPT, adversarial) import
from here. This module provides:

- ``BenchmarkConfig``  — engine + run configuration (dataclass).
- ``RequestRecord``    — a single (prompt, output_len) pair.
- ``BenchmarkResult``  — aggregated latency/throughput metrics.
- ``replay_trace()``   — drives ``llm.generate()`` in arrival-ordered batches
  and measures per-request latency.
- ``write_result()``   — writes a ``BenchmarkResult`` to JSON.
- ``load_trace_from_jsonl()`` / ``save_trace_to_jsonl()`` — trace I/O.

The harness is intentionally engine-agnostic: ``replay_trace()`` accepts any
object with a ``generate(prompts, sampling_params)`` interface so that
``validate_harness.py`` can inject a mock without loading a real model.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vllm import SamplingParams


@dataclass
class BenchmarkConfig:
    """Configuration shared by all benchmark entry points."""

    model: str
    scheduler_cls: str = (
        "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
    )
    baseline_scheduler_cls: str = "vllm.v1.core.sched.scheduler.Scheduler"
    max_model_len: int = 4096
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 4096
    # Number of repetitions for statistical stability.
    num_iters: int = 3
    # Token-budget for each generated response.
    max_new_tokens: int = 64
    # Seed for reproducible prompt generation.
    seed: int = 42
    # Extra engine kwargs forwarded verbatim to EngineArgs.
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestRecord:
    """A single workload request."""

    prompt: str
    output_len: int
    # Metadata preserved across trace serialisation.
    group_id: str = ""
    priority: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated result for one scheduler / scenario combination."""

    scenario: str
    scheduler_cls: str
    num_requests: int
    # Latency distribution (seconds).
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    # End-to-end wall time for the full trace (seconds).
    total_wall_time_s: float
    # Derived throughput.
    throughput_req_s: float
    # Raw per-request latencies for downstream analysis.
    latencies_s: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineProtocol(Protocol):
    """Minimal interface expected from the engine by replay_trace()."""

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        **kwargs: Any,
    ) -> list[Any]: ...


# ---------------------------------------------------------------------------
# Core replay driver
# ---------------------------------------------------------------------------


def replay_trace(
    records: list[RequestRecord],
    engine: EngineProtocol,
    *,
    max_new_tokens: int = 64,
    batch_size: int | None = None,
) -> list[float]:
    """Submit ``records`` to ``engine`` and return per-request latencies (s).

    Requests are submitted in the order they appear in ``records`` (which
    should already be sorted by intended arrival order).  When ``batch_size``
    is None every record is submitted in a single ``generate()`` call,
    mirroring how ``benchmark_prefix_caching.py`` operates.  Pass an integer
    to submit in fixed-size batches (useful when simulating bursty arrivals).

    Returns a flat list of per-request elapsed times in arrival order.
    """
    if not records:
        return []

    if batch_size is None:
        batch_size = len(records)

    latencies: list[float] = []
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        prompts = [r.prompt for r in batch]
        sampling_params_list = [
            SamplingParams(max_tokens=r.output_len or max_new_tokens, temperature=0.0)
            for r in batch
        ]
        t0 = time.perf_counter()
        engine.generate(prompts, sampling_params_list)
        elapsed = time.perf_counter() - t0
        # Distribute elapsed time equally across the batch.  A per-token TTFT
        # breakdown would require the async engine; this gives a fair
        # throughput-level comparison.
        per_req = elapsed / len(batch)
        latencies.extend([per_req] * len(batch))

    return latencies


def aggregate(
    latencies: list[float],
    scenario: str,
    scheduler_cls: str,
) -> BenchmarkResult:
    """Convert a flat list of latencies into a ``BenchmarkResult``."""
    n = len(latencies)
    if n == 0:
        raise ValueError("Cannot aggregate an empty latency list.")
    sorted_lats = sorted(latencies)
    total = sum(latencies)

    def _percentile(sorted_list: list[float], p: float) -> float:
        idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
        return sorted_list[idx]

    return BenchmarkResult(
        scenario=scenario,
        scheduler_cls=scheduler_cls,
        num_requests=n,
        mean_latency_s=statistics.mean(latencies),
        p50_latency_s=_percentile(sorted_lats, 50),
        p95_latency_s=_percentile(sorted_lats, 95),
        p99_latency_s=_percentile(sorted_lats, 99),
        total_wall_time_s=total,
        throughput_req_s=n / total if total > 0 else 0.0,
        latencies_s=latencies,
    )


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def write_result(result: BenchmarkResult, output_path: Path) -> None:
    """Write a ``BenchmarkResult`` to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[harness] Result written to {output_path}")


# ---------------------------------------------------------------------------
# Trace serialisation
# ---------------------------------------------------------------------------


def save_trace_to_jsonl(records: list[RequestRecord], path: Path) -> None:
    """Serialise a list of ``RequestRecord`` objects to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    print(f"[harness] Trace saved to {path} ({len(records)} records)")


def load_trace_from_jsonl(path: Path) -> list[RequestRecord]:
    """Load ``RequestRecord`` objects from a JSONL file."""
    records: list[RequestRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records.append(RequestRecord(**d))
    return records


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_results(
    baseline: BenchmarkResult,
    affinity: BenchmarkResult,
) -> dict[str, float]:
    """Return a dict of improvement ratios (affinity vs baseline).

    Positive values mean affinity is better:
    - ``latency_improvement_pct``: % reduction in mean latency.
    - ``throughput_improvement_pct``: % increase in throughput.
    """
    lat_delta = baseline.mean_latency_s - affinity.mean_latency_s
    lat_pct = (
        lat_delta / baseline.mean_latency_s * 100
        if baseline.mean_latency_s > 0
        else 0.0
    )
    tpt_delta = affinity.throughput_req_s - baseline.throughput_req_s
    tpt_pct = (
        tpt_delta / baseline.throughput_req_s * 100
        if baseline.throughput_req_s > 0
        else 0.0
    )
    return {
        "latency_improvement_pct": lat_pct,
        "throughput_improvement_pct": tpt_pct,
        "baseline_mean_latency_s": baseline.mean_latency_s,
        "affinity_mean_latency_s": affinity.mean_latency_s,
        "baseline_throughput_req_s": baseline.throughput_req_s,
        "affinity_throughput_req_s": affinity.throughput_req_s,
    }
