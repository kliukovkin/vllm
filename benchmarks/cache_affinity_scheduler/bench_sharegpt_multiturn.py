# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShareGPT multi-turn benchmark.

Loads a ShareGPT trace (``--dataset-path``) and constructs multi-turn
conversation prompts where each turn appends the previous turns as context.
Multi-turn conversations naturally exhibit prefix sharing: turns 2..N of a
conversation share the history of turns 1..N-1.

The trace is grouped by conversation so that related turns are submitted
together, maximising the observable cache-affinity benefit.

Usage::

    python -m benchmarks.cache_affinity_scheduler.bench_sharegpt_multiturn \\
        --model facebook/opt-125m \\
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-conversations 32
"""

from __future__ import annotations

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
    save_trace_to_jsonl,
    write_result,
)
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _load_sharegpt(
    path: Path,
    num_conversations: int,
    max_turns: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestRecord]:
    """Load ShareGPT conversations and build multi-turn prompt records.

    Each turn in a conversation is represented as a ``RequestRecord`` whose
    prompt is the full conversation history up to that turn.  This naturally
    creates long shared prefixes between consecutive turns.
    """
    with open(path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    # Filter to conversations with at least 2 turns.
    valid = [
        conv
        for conv in data
        if isinstance(conv.get("conversations"), list)
        and len(conv["conversations"]) >= 2
    ]
    rng.shuffle(valid)
    selected = valid[:num_conversations]

    records: list[RequestRecord] = []
    for conv_idx, conv in enumerate(selected):
        turns = conv["conversations"][:max_turns]
        history = ""
        for i, turn in enumerate(turns):
            role = turn.get("from", "human")
            text = turn.get("value", "")
            history += f"[{role.upper()}] {text}\n"
            if role in ("gpt", "assistant") and i > 0:
                # Only submit on assistant turns (the model response turns).
                records.append(
                    RequestRecord(
                        prompt=history.strip(),
                        output_len=max_new_tokens,
                        group_id=f"conv{conv_idx}",
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
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to ShareGPT JSON dataset.",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=32,
        help="Number of conversations to sample.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=4, help="Max turns per conversation."
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--save-trace",
        type=Path,
        default=None,
        help="If set, save the trace to this JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/cache_affinity_scheduler/results"),
    )
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    records = _load_sharegpt(
        path=args.dataset_path,
        num_conversations=args.num_conversations,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"[sharegpt] {len(records)} turn-requests from "
        f"{args.num_conversations} conversations"
    )

    if args.save_trace:
        save_trace_to_jsonl(records, args.save_trace)

    affinity_cls = "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
    baseline_cls = "vllm.v1.core.sched.scheduler.Scheduler"

    baseline = _run_once(
        records, engine_args, baseline_cls, args.max_new_tokens, "sharegpt_baseline"
    )
    affinity = _run_once(
        records, engine_args, affinity_cls, args.max_new_tokens, "sharegpt_affinity"
    )

    write_result(baseline, args.output_dir / "sharegpt_baseline.json")
    write_result(affinity, args.output_dir / "sharegpt_affinity.json")

    cmp = compare_results(baseline, affinity)
    print(
        f"[sharegpt] latency improvement: "
        f"{cmp['latency_improvement_pct']:+.1f}%  "
        f"throughput improvement: "
        f"{cmp['throughput_improvement_pct']:+.1f}%"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
