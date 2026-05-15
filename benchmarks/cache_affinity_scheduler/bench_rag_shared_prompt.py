# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAG shared-system-prompt benchmark.

Simulates a Retrieval-Augmented Generation workload where every request shares
a long system prompt (``--system-prompt-len`` tokens) followed by a short,
unique user query.  This is the canonical use-case for cache-affinity
scheduling: the system prompt should be KV-cached after the first request and
reused for all subsequent ones.

Usage::

    python -m benchmarks.cache_affinity_scheduler.bench_rag_shared_prompt \\
        --model facebook/opt-125m \\
        --system-prompt-len 512 \\
        --num-queries 32 \\
        --query-len 32
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from benchmarks.cache_affinity_scheduler.harness import (
    BenchmarkResult,
    RequestRecord,
    aggregate,
    compare_results,
    replay_trace,
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
    # Build one long shared system prompt.
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


def _run_once(
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
        "--output-dir",
        type=Path,
        default=Path("benchmarks/cache_affinity_scheduler/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
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

    affinity_cls = "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
    baseline_cls = "vllm.v1.core.sched.scheduler.Scheduler"

    baseline = _run_once(
        records, engine_args, baseline_cls, args.max_new_tokens, "rag_baseline"
    )
    affinity = _run_once(
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
