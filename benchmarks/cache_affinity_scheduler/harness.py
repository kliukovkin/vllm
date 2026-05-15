# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness utilities for CacheAffinityScheduler benchmarks.

All four benchmark scripts (synthetic, RAG, ShareGPT, adversarial) import
from here. This module provides:

- ``BenchmarkConfig``  — engine + run configuration (dataclass).
- ``RequestRecord``    — a single (prompt, output_len) pair with an optional
  ``arrival_time_offset_s`` for online/QPS-throttled replay.
- ``BenchmarkResult``  — aggregated latency/throughput metrics.
- ``replay_trace()``   — offline batch replay (all requests submitted at once).
- ``replay_trace_online()`` — online step-loop replay honouring per-request
  arrival times; requires a ``vllm.LLM`` instance.
- ``assign_poisson_arrivals()`` — compute Poisson inter-arrival offsets and
  write them onto ``RequestRecord.arrival_time_offset_s``.
- ``write_result()``   — writes a ``BenchmarkResult`` to JSON.
- ``load_trace_from_jsonl()`` / ``save_trace_to_jsonl()`` — trace I/O.

The offline ``replay_trace()`` is intentionally engine-agnostic: it accepts
any object with a ``generate(prompts, sampling_params)`` interface so that
``validate_harness.py`` can inject a mock without loading a real model.

The online ``replay_trace_online()`` requires a real ``vllm.LLM`` instance
and accesses ``llm.llm_engine`` directly to drive the step loop.
"""

from __future__ import annotations

import json
import random
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
    # Seconds after benchmark start when this request should arrive.
    # 0.0 means "submit immediately" (offline / batch mode).
    arrival_time_offset_s: float = 0.0


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
# Engine protocol (offline batch mode)
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
# Arrival-time helpers
# ---------------------------------------------------------------------------


def assign_poisson_arrivals(
    records: list[RequestRecord],
    *,
    qps: float,
    seed: int = 42,
) -> None:
    """Assign Poisson inter-arrival offsets to each record in-place.

    After this call, ``record.arrival_time_offset_s`` holds the time (in
    seconds after benchmark start) at which that request should be submitted.
    """
    rng = random.Random(seed)
    t = 0.0
    for record in records:
        record.arrival_time_offset_s = t
        t += rng.expovariate(qps)


# ---------------------------------------------------------------------------
# Core replay drivers
# ---------------------------------------------------------------------------


def replay_trace(
    records: list[RequestRecord],
    engine: EngineProtocol,
    *,
    max_new_tokens: int = 64,
    batch_size: int | None = None,
) -> list[float]:
    """Submit ``records`` to ``engine`` and return per-request latencies (s).

    Offline / batch mode: all requests submitted simultaneously (arrival times
    ignored).  Suitable for quick iteration tests and the adversarial / no-QPS
    benchmarks.  Returns a flat list of per-request elapsed times.
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
        per_req = elapsed / len(batch)
        latencies.extend([per_req] * len(batch))

    return latencies


def replay_trace_online(
    records: list[RequestRecord],
    llm: Any,  # vllm.LLM
    *,
    max_new_tokens: int = 64,
) -> tuple[list[float], float]:
    """Online step-loop replay honouring ``arrival_time_offset_s``.

    Submits requests to the underlying v1 LLMEngine one by one as their
    scheduled arrival time approaches, then drives the engine step loop until
    all requests complete.  This creates a real waiting queue whose depth and
    composition vary over time — the scenario where cache-affinity reordering
    provides genuine value.

    Returns ``(per_request_latencies_s, total_wall_time_s)``.  Each latency
    is measured from the request's target arrival time to its completion.
    ``total_wall_time_s`` is the actual wall-clock span from benchmark start
    to the last completion, suitable for throughput calculation.
    """
    if not records:
        return [], 0.0

    engine = llm.llm_engine
    # Sort by intended arrival time (defensive).
    records_sorted = sorted(records, key=lambda r: r.arrival_time_offset_s)
    total = len(records_sorted)

    benchmark_start = time.monotonic()
    # Map internal (randomised) request_id → position in records_sorted
    id_to_idx: dict[str, int] = {}
    # Intended arrival offset per index (used as latency start)
    submit_offsets: dict[int, float] = {}
    # Completed latencies keyed by position
    latencies: dict[int, float] = {}
    next_idx = 0

    while next_idx < total or engine.has_unfinished_requests():
        now = time.monotonic() - benchmark_start

        # Submit all requests whose arrival window has opened.
        while (
            next_idx < total and records_sorted[next_idx].arrival_time_offset_s <= now
        ):
            r = records_sorted[next_idx]
            sp = SamplingParams(
                max_tokens=r.output_len or max_new_tokens,
                temperature=0.0,
            )
            # Pass the monotonic arrival_time so the scheduler's wait_s
            # computation (which uses time.monotonic()) is correct.
            actual_arrival = benchmark_start + r.arrival_time_offset_s
            req_id = engine.add_request(
                f"bench_{next_idx}",
                r.prompt,
                sp,
                arrival_time=actual_arrival,
            )
            id_to_idx[req_id] = next_idx
            submit_offsets[next_idx] = r.arrival_time_offset_s
            next_idx += 1

        # If the engine has nothing to do and no request has arrived yet,
        # sleep until just before the next arrival to avoid spinning.
        if not engine.has_unfinished_requests() and next_idx < total:
            next_arrival = records_sorted[next_idx].arrival_time_offset_s
            sleep_s = next_arrival - (time.monotonic() - benchmark_start)
            if sleep_s > 0.001:
                time.sleep(sleep_s * 0.9)
            continue

        step_outputs = engine.step()
        step_end = time.monotonic() - benchmark_start

        for out in step_outputs:
            if out.finished and out.request_id in id_to_idx:
                idx = id_to_idx.pop(out.request_id)
                latencies[idx] = step_end - submit_offsets[idx]

    total_wall_time_s = time.monotonic() - benchmark_start
    per_req_latencies = [latencies.get(i, 0.0) for i in range(total)]
    return per_req_latencies, total_wall_time_s


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    latencies: list[float],
    scenario: str,
    scheduler_cls: str,
) -> BenchmarkResult:
    """Convert a flat list of offline latencies into a ``BenchmarkResult``.

    In offline mode the sum of per-request latencies equals the wall-clock
    time, so ``throughput_req_s = n / sum(latencies)``.
    """
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


def aggregate_online(
    latencies: list[float],
    total_wall_time_s: float,
    scenario: str,
    scheduler_cls: str,
) -> BenchmarkResult:
    """Convert online-mode per-request latencies into a ``BenchmarkResult``.

    In online mode per-request latencies overlap (concurrent execution), so
    the wall-clock time is passed separately and used for throughput.
    """
    n = len(latencies)
    if n == 0:
        raise ValueError("Cannot aggregate an empty latency list.")
    sorted_lats = sorted(latencies)

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
        total_wall_time_s=total_wall_time_s,
        throughput_req_s=n / total_wall_time_s if total_wall_time_s > 0 else 0.0,
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
