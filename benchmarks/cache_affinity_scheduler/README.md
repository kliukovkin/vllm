# CacheAffinityScheduler Benchmarks

Benchmark harness for
[`CacheAffinityScheduler`](../../vllm/v1/core/sched/cache_affinity_scheduler.py),
a vLLM V1 scheduler plugin that reorders the waiting queue by cached-prefix
length before each scheduling iteration.

## Structure

```text
benchmarks/cache_affinity_scheduler/
├── harness.py                   # Shared utilities (dataclasses, replay, I/O)
├── bench_synthetic.py           # Synthetic same-prefix burst workload
├── bench_rag_shared_prompt.py   # RAG workload with long shared system prompt
├── bench_sharegpt_multiturn.py  # ShareGPT multi-turn conversation workload
├── bench_adversarial.py         # Regression gate: unique-prompt workload
├── validate_harness.py          # CPU-only dry-run (no model download needed)
└── traces/                      # Pre-built JSONL traces (gitignored except .gitkeep)
    └── synthetic/
```

## Quick Start

### Validate without a model (CPU dry-run)

```bash
python -m benchmarks.cache_affinity_scheduler.validate_harness
```

Expected: exits 0 with `ALL CHECKS PASSED`.

### Synthetic workload (same-prefix bursts)

```bash
python -m benchmarks.cache_affinity_scheduler.bench_synthetic \
    --model facebook/opt-125m \
    --num-groups 8 \
    --group-size 4 \
    --prefix-len 512
```

Sweep bucket-edge configurations:

```bash
python -m benchmarks.cache_affinity_scheduler.bench_synthetic \
    --model facebook/opt-125m \
    --sweep-bucket-edges "4,16,64,256" "2,8,32,128" "1,4,16,64" \
    --num-groups 16 --group-size 8 --prefix-len 512
```

### RAG shared system prompt

```bash
python -m benchmarks.cache_affinity_scheduler.bench_rag_shared_prompt \
    --model facebook/opt-125m \
    --system-prompt-len 512 \
    --num-queries 64
```

### ShareGPT multi-turn

Requires a local copy of the ShareGPT dataset
(`ShareGPT_V3_unfiltered_cleaned_split.json`).

```bash
python -m benchmarks.cache_affinity_scheduler.bench_sharegpt_multiturn \
    --model facebook/opt-125m \
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-conversations 64
```

### Adversarial regression gate

```bash
python -m benchmarks.cache_affinity_scheduler.bench_adversarial \
    --model facebook/opt-125m \
    --num-requests 64 \
    --regression-threshold-pct 5.0
# Exits 0 (PASS) or 1 (FAIL).
```

## Output

Each benchmark writes JSON result files to `--output-dir`
(default: `benchmarks/cache_affinity_scheduler/results/`).

```json
{
  "scenario": "synthetic_affinity_default",
  "scheduler_cls": "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler",
  "num_requests": 32,
  "mean_latency_s": 0.042,
  "p50_latency_s": 0.041,
  "p95_latency_s": 0.058,
  "p99_latency_s": 0.063,
  "total_wall_time_s": 1.34,
  "throughput_req_s": 23.9,
  "latencies_s": [...]
}
```

## Loading the Scheduler

All benchmark scripts forward `--scheduler-cls` to the engine automatically
via `EngineArgs.add_cli_args()`.  You can override the scheduler manually:

```bash
python -m benchmarks.cache_affinity_scheduler.bench_synthetic \
    --model facebook/opt-125m \
    --scheduler-cls vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler
```

## Tuning Parameters

| CLI flag | Config field | Default | Description |
| --- | --- | --- | --- |
| `--cache-affinity-enabled` | `cache_affinity_enabled` | `true` | Enable/disable reordering |
| `--cache-affinity-max-wait-s` | `cache_affinity_max_wait_s` | `0.2` | Anti-starvation deadline (s) |
| `--cache-affinity-min-blocks` | `cache_affinity_min_blocks` | `2` | Min cached blocks to be "warm" |
| `--cache-affinity-bucket-edges` | `cache_affinity_bucket_edges` | `4,16,64,256` | Bucket boundaries |
