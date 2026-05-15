# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adversarial regression gate: unique-prompt workload.

Measures performance when every request has a fully unique prompt (no prefix
sharing).  CacheAffinityScheduler should not regress vs the baseline in this
case because all requests will score 0 cached blocks and the sort order will
fall back to arrival_time (FCFS), matching the default scheduler.

Exit code: 0 if the affinity scheduler throughput is within
``--regression-threshold-pct`` of the baseline; 1 otherwise.  This makes the
script suitable as a CI regression gate.

Usage::

    python -m benchmarks.cache_affinity_scheduler.bench_adversarial \\
        --model facebook/opt-125m \\
        --num-requests 32 \\
        --prompt-len 64 \\
        --regression-threshold-pct 5.0
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
    num_requests: int,
    prompt_len: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestRecord]:
    """Build a trace where every prompt is fully unique (no shared prefix)."""
    rng = random.Random(seed)
    records: list[RequestRecord] = []
    for i in range(num_requests):
        tokens = [str(rng.randint(1, _VOCAB_SIZE - 1)) for _ in range(prompt_len)]
        records.append(
            RequestRecord(
                prompt=" ".join(tokens),
                output_len=max_new_tokens,
                group_id=f"unique_{i}",
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


def main() -> int:
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--num-requests", type=int, default=32, help="Number of unique-prompt requests."
    )
    parser.add_argument(
        "--prompt-len", type=int, default=64, help="Prompt length in tokens (approx)."
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--regression-threshold-pct",
        type=float,
        default=5.0,
        help="Max allowable throughput regression (%).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/cache_affinity_scheduler/results"),
    )
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    records = _build_trace(
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"[adversarial] {len(records)} unique-prompt requests, "
        f"prompt_len≈{args.prompt_len}"
    )

    affinity_cls = "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
    baseline_cls = "vllm.v1.core.sched.scheduler.Scheduler"

    baseline = _run_once(
        records, engine_args, baseline_cls, args.max_new_tokens, "adversarial_baseline"
    )
    affinity = _run_once(
        records, engine_args, affinity_cls, args.max_new_tokens, "adversarial_affinity"
    )

    write_result(baseline, args.output_dir / "adversarial_baseline.json")
    write_result(affinity, args.output_dir / "adversarial_affinity.json")

    cmp = compare_results(baseline, affinity)
    regression_pct = -cmp["throughput_improvement_pct"]  # negative = regression

    print(
        f"[adversarial] throughput improvement: "
        f"{cmp['throughput_improvement_pct']:+.1f}% "
        f"(threshold: -{args.regression_threshold_pct:.1f}%)"
    )

    if regression_pct > args.regression_threshold_pct:
        print(
            f"[adversarial] FAIL: throughput regressed by "
            f"{regression_pct:.1f}% > {args.regression_threshold_pct:.1f}%"
        )
        return 1

    print("[adversarial] PASS: within regression threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
