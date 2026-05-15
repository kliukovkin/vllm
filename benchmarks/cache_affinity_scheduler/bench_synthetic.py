# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic cache-affinity benchmark: same-prefix bursts.

Generates ``--num-groups`` groups of ``--group-size`` requests that all share
the same long prefix (``--prefix-len`` tokens).  Each group is a burst of
requests that should benefit maximally from cache affinity.

Supports ``--sweep-bucket-edges`` to run across multiple bucket-edge
configurations and dump a comparison table.

Usage::

    python -m benchmarks.cache_affinity_scheduler.bench_synthetic \\
        --model facebook/opt-125m \\
        --num-groups 8 \\
        --group-size 4 \\
        --prefix-len 512

    # Sweep bucket edges:
    python -m benchmarks.cache_affinity_scheduler.bench_synthetic \\
        --model facebook/opt-125m \\
        --sweep-bucket-edges "4,16,64,256" "2,8,32,128" \\
        --num-groups 8 --group-size 4 --prefix-len 512
"""

from __future__ import annotations

import argparse
import json
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

_VOCAB_SIZE = 32000  # conservative; actual vocab checked against tokeniser


def _build_trace(
    *,
    num_groups: int,
    group_size: int,
    prefix_len: int,
    suffix_len: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestRecord]:
    """Build a synthetic trace with ``num_groups`` same-prefix bursts."""
    rng = random.Random(seed)
    records: list[RequestRecord] = []
    for g in range(num_groups):
        # Each group shares the same prefix (repeating token id) plus unique suffixes.
        prefix_tok = rng.randint(1, _VOCAB_SIZE - 1)
        prefix_str = f"[PREFIX_{g}] " + " ".join([str(prefix_tok)] * prefix_len)
        for i in range(group_size):
            suffix = " ".join(
                [str(rng.randint(1, _VOCAB_SIZE - 1)) for _ in range(suffix_len)]
            )
            prompt = prefix_str + " [SEP] " + suffix
            records.append(
                RequestRecord(
                    prompt=prompt,
                    output_len=max_new_tokens,
                    group_id=f"g{g}",
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


def _parse_bucket_edges(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    parser = FlexibleArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--num-groups", type=int, default=8, help="Number of request groups (bursts)."
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Requests per group (all share the same prefix).",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=256,
        help="Shared prefix length in tokens (approximate).",
    )
    parser.add_argument(
        "--suffix-len", type=int, default=32, help="Unique suffix length per request."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Max output tokens per request."
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Repetitions for statistical stability.",
    )
    parser.add_argument(
        "--sweep-bucket-edges",
        nargs="+",
        default=None,
        metavar="EDGES",
        help="Comma-separated bucket edge sets to sweep, e.g. "
        '"4,16,64,256" "2,8,32,128".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/cache_affinity_scheduler/results"),
        help="Directory to write JSON result files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    records = _build_trace(
        num_groups=args.num_groups,
        group_size=args.group_size,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"[synthetic] {len(records)} requests, "
        f"{args.num_groups} groups x {args.group_size}, "
        f"prefix_len≈{args.prefix_len}"
    )

    bucket_edge_sets: list[tuple[int, ...] | None]
    if args.sweep_bucket_edges:
        bucket_edge_sets = [_parse_bucket_edges(s) for s in args.sweep_bucket_edges]
    else:
        bucket_edge_sets = [None]

    all_comparisons: list[dict] = []
    for edges in bucket_edge_sets:
        if edges is not None:
            engine_args.cache_affinity_bucket_edges = edges
            tag = f"edges={'_'.join(str(e) for e in edges)}"
        else:
            tag = "default"

        affinity_cls = (
            "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
        )
        baseline_cls = "vllm.v1.core.sched.scheduler.Scheduler"

        baseline = _run_once(
            records,
            engine_args,
            baseline_cls,
            args.max_new_tokens,
            f"synthetic_baseline_{tag}",
        )
        affinity = _run_once(
            records,
            engine_args,
            affinity_cls,
            args.max_new_tokens,
            f"synthetic_affinity_{tag}",
        )

        write_result(baseline, args.output_dir / f"synthetic_baseline_{tag}.json")
        write_result(affinity, args.output_dir / f"synthetic_affinity_{tag}.json")

        cmp = compare_results(baseline, affinity)
        cmp["tag"] = tag
        all_comparisons.append(cmp)
        print(
            f"[{tag}] latency improvement: "
            f"{cmp['latency_improvement_pct']:+.1f}%  "
            f"throughput improvement: "
            f"{cmp['throughput_improvement_pct']:+.1f}%"
        )

    if len(all_comparisons) > 1:
        sweep_path = args.output_dir / "synthetic_sweep.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(all_comparisons, f, indent=2)
        print(f"[synthetic] Sweep summary → {sweep_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
