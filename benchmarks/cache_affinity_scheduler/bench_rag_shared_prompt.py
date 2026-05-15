# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAG shared-system-prompt benchmark.

Simulates a Retrieval-Augmented Generation workload where every request shares
a long system prompt (``--system-prompt-len`` tokens) followed by a short,
unique user query.  This is the canonical use-case for cache-affinity
scheduling: the system prompt should be KV-cached after the first request and
reused for all subsequent ones.

**Offline mode** (no ``--qps``, default):
  Both baseline and affinity schedulers are run internally.  All requests are
  submitted at once (batch mode).  Produces ``rag_baseline.json`` and
  ``rag_affinity.json`` in ``--output-dir``.

**Online mode** (``--qps N``):
  A single scheduler run honouring Poisson inter-arrival times.  The
  scheduler is taken from ``--scheduler-cls`` (or ``engine_args`` default).
  Produces a single ``result.json`` in ``--output-dir``.  Use an external
  shell loop to run baseline and affinity schedulers separately.

Usage::

    # Offline (both schedulers, batch mode):
    python -m benchmarks.cache_affinity_scheduler.bench_rag_shared_prompt \\
        --model facebook/opt-125m \\
        --system-prompt-len 512 \\
        --num-queries 32 \\
        --query-len 32

    # Online (single scheduler, QPS-throttled):
    python -m benchmarks.cache_affinity_scheduler.bench_rag_shared_prompt \\
        --model facebook/opt-125m \\
        --system-prompt-len 512 \\
        --num-queries 32 \\
        --query-len 32 \\
        --qps 10 \\
        --scheduler-cls \
            vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from benchmarks.cache_affinity_scheduler.harness import (
    BenchmarkResult,
    RequestRecord,
    aggregate,
    aggregate_online,
    assign_poisson_arrivals,
    compare_results,
    replay_trace,
    replay_trace_online,
    write_result,
)
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

_VOCAB_SIZE = 32000


def _build_trace(
    *,
    system_prompt_len: int,
    num_queries: int,
    query_len: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestRecord]:
    rng = random.Random(seed)
    shared_prefix = "[SYSTEM] " + " ".join(
        [str(rng.randint(1, _VOCAB_SIZE - 1)) for _ in range(system_prompt_len)]
    )
    records: list[RequestRecord] = []
    for i in range(num_queries):
        query = " ".join(
            [str(rng.randint(1, _VOCAB_SIZE - 1)) for _ in range(query_len)]
        )
        records.append(
            RequestRecord(
                prompt=f"{shared_prefix}\n[USER] {query}\n[ASSISTANT]",
                output_len=max_new_tokens,
                group_id="rag",
            )
        )
    return records


def _run_once_offline(
    records: list[RequestRecord],
    engine_args: EngineArgs,
    scheduler_cls: str,
    max_new_tokens: int,
    scenario: str,
) -> BenchmarkResult:
    engine_args.scheduler_cls = scheduler_cls
    llm = LLM.from_engine_args(engine_args)
    latencies = replay_trace(records, llm, max_new_tokens=max_new_tokens)
    del llm
    return aggregate(latencies, scenario, scheduler_cls)


def _run_once_online(
    records: list[RequestRecord],
    engine_args: EngineArgs,
    max_new_tokens: int,
    scenario: str,
    qps: float,
    seed: int,
) -> BenchmarkResult:
    assign_poisson_arrivals(records, qps=qps, seed=seed)
    llm = LLM.from_engine_args(engine_args)
    latencies, wall_time = replay_trace_online(
        records, llm, max_new_tokens=max_new_tokens
    )
    del llm
    return aggregate_online(
        latencies, wall_time, scenario, engine_args.scheduler_cls or ""
    )


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--system-prompt-len",
        type=int,
        default=512,
        help="Shared system-prompt length in tokens (approx).",
    )
    parser.add_argument(
        "--num-queries", type=int, default=32, help="Number of user queries."
    )
    parser.add_argument(
        "--query-len",
        type=int,
        default=32,
        help="Unique query length in tokens (approx).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help=(
            "Target request rate (req/s) for online Poisson-arrival mode. "
            "When set, runs a single scheduler (from --scheduler-cls) with "
            "real inter-arrival delays. When unset, runs both baseline and "
            "affinity schedulers in offline batch mode."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/cache_affinity_scheduler/results"),
    )
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    records = _build_trace(
        system_prompt_len=args.system_prompt_len,
        num_queries=args.num_queries,
        query_len=args.query_len,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"[rag] {len(records)} requests, "
        f"system_prompt_len≈{args.system_prompt_len}, "
        f"query_len≈{args.query_len}"
    )

    if args.qps is not None:
        # Online single-scheduler mode.
        sched_name = (engine_args.scheduler_cls or "baseline").split(".")[-1].lower()
        scenario = f"rag_online_{sched_name}"
        result = _run_once_online(
            records,
            engine_args,
            args.max_new_tokens,
            scenario,
            qps=args.qps,
            seed=args.seed,
        )
        write_result(result, args.output_dir / "result.json")
        print(
            f"[rag] qps={args.qps} scheduler={sched_name} "
            f"mean_lat={result.mean_latency_s:.3f}s "
            f"p99_lat={result.p99_latency_s:.3f}s "
            f"throughput={result.throughput_req_s:.3f} req/s"
        )
    else:
        # Offline both-scheduler mode (backward compatibility).
        affinity_cls = (
            "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
        )
        baseline_cls = "vllm.v1.core.sched.scheduler.Scheduler"

        baseline = _run_once_offline(
            records, engine_args, baseline_cls, args.max_new_tokens, "rag_baseline"
        )
        affinity = _run_once_offline(
            records, engine_args, affinity_cls, args.max_new_tokens, "rag_affinity"
        )

        write_result(baseline, args.output_dir / "rag_baseline.json")
        write_result(affinity, args.output_dir / "rag_affinity.json")

        cmp = compare_results(baseline, affinity)
        print(
            f"[rag] latency improvement: {cmp['latency_improvement_pct']:+.1f}%  "
            f"throughput improvement: {cmp['throughput_improvement_pct']:+.1f}%"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
