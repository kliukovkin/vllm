# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CacheAffinityScheduler and CacheAffinityRequestQueue."""

import time
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.cache_affinity_scheduler import (
    CacheAffinityRequestQueue,
    CacheAffinityScheduler,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_block_hasher_initialized = False


def _ensure_block_hasher_init() -> None:
    global _block_hasher_initialized
    if not _block_hasher_initialized:
        init_none_hash(sha256)
        _block_hasher_initialized = True


def make_request(
    req_id: str,
    num_tokens: int = 10,
    priority: int = 0,
    arrival_time: float | None = None,
    max_tokens: int = 16,
    block_size: int = 16,
) -> Request:
    """Create a minimal Request for testing."""
    _ensure_block_hasher_init()
    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    req = Request(
        request_id=req_id,
        prompt_token_ids=[req_id.__hash__() % 1000] * num_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=priority,
        arrival_time=arrival_time if arrival_time is not None else time.monotonic(),
        block_hasher=block_hasher,
    )
    return req


def create_cas_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    enable_prefix_caching: bool = False,
    policy: str = "fcfs",
    cache_affinity_enabled: bool = True,
    cache_affinity_max_wait_s: float = 0.2,
    cache_affinity_min_blocks: int = 2,
    cache_affinity_bucket_edges: tuple = (4, 16, 64, 256),
    cache_affinity_batch_guard_threshold_s: float = 0.01,
) -> CacheAffinityScheduler:
    """Create a CacheAffinityScheduler under test."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=policy,
        cache_affinity_enabled=cache_affinity_enabled,
        cache_affinity_max_wait_s=cache_affinity_max_wait_s,
        cache_affinity_min_blocks=cache_affinity_min_blocks,
        cache_affinity_bucket_edges=cache_affinity_bucket_edges,
        cache_affinity_batch_guard_threshold_s=cache_affinity_batch_guard_threshold_s,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    cache_config.num_gpu_blocks = num_blocks
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return CacheAffinityScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _mock_get_computed_blocks(scheduler: CacheAffinityScheduler, scores: dict) -> Mock:
    """Replace kv_cache_manager.get_computed_blocks with a mock that returns
    controlled (KVCacheBlocks, num_cached_tokens) pairs based on req.request_id.
    ``scores`` maps request_id -> num_cached_tokens (integer).
    """
    block_size = scheduler.block_size

    def _side_effect(req: Request):
        n = scores.get(req.request_id, 0)
        mock_blocks = MagicMock()
        mock_blocks.get_block_ids.return_value = (list(range(n // block_size)),)
        return mock_blocks, n

    mock_fn = Mock(side_effect=_side_effect)
    scheduler.kv_cache_manager.get_computed_blocks = mock_fn
    return mock_fn


def _drain_sortable(scheduler: CacheAffinityScheduler) -> list[str]:
    """Return the request_ids in sortable-deque order without popping from the
    queue (non-destructive peek of sort result)."""
    assert isinstance(scheduler.waiting, CacheAffinityRequestQueue)
    return [r.request_id for r in scheduler.waiting.iter_sortable()]


def _drain_queue(scheduler: CacheAffinityScheduler) -> list[str]:
    """Pop all requests from the waiting queue and return their IDs in order."""
    result = []
    while scheduler.waiting:
        result.append(scheduler.waiting.pop_request().request_id)
    return result


# ---------------------------------------------------------------------------
# Test 5.1: empty queue — no-op
# ---------------------------------------------------------------------------


def test_empty_queue_noop():
    """schedule() with an empty waiting queue must not crash and must not call
    _reorder_waiting (verified via sort-latency counter)."""
    sched = create_cas_scheduler()
    before_samples = len(sched._stat_sort_us_samples)
    # No requests added — schedule() should produce an empty output.
    output = sched.schedule()
    assert output is not None
    # No sort should have been triggered (queue length 0 or 1 → skip).
    assert len(sched._stat_sort_us_samples) == before_samples


# ---------------------------------------------------------------------------
# Test 5.2: single request — no reorder
# ---------------------------------------------------------------------------


def test_single_request_noop():
    """schedule() with exactly one waiting request must not trigger reordering."""
    sched = create_cas_scheduler()
    req = make_request("r0")
    sched.add_request(req)
    before_samples = len(sched._stat_sort_us_samples)
    sched.schedule()
    assert len(sched._stat_sort_us_samples) == before_samples


# ---------------------------------------------------------------------------
# Test 5.3: cache-warm request is promoted to front
# ---------------------------------------------------------------------------


def test_reorder_promotes_cache_warm():
    """Three waiting requests; the one with the most cached blocks should move
    to the front after _reorder_waiting()."""
    sched = create_cas_scheduler(cache_affinity_min_blocks=1)
    now = time.monotonic()
    # Use arrival times spread > 10 ms (batch-guard threshold) so the guard
    # does not suppress the reorder.
    req_a = make_request("a", arrival_time=now - 0.100)  # earliest, cold
    req_b = make_request("b", arrival_time=now - 0.000)  # latest, warm (10 blocks)
    req_c = make_request("c", arrival_time=now - 0.050)  # middle, cold

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": 0, "b": block_size * 10, "c": 0})

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    assert order[0] == "b", f"Expected 'b' at front, got {order}"
    assert set(order) == {"a", "b", "c"}
    # a and c both have score 0; a has earlier arrival so a < c
    assert order[1] == "a"
    assert order[2] == "c"


# ---------------------------------------------------------------------------
# Test 5.4: min_blocks threshold — one cached block does not promote
# ---------------------------------------------------------------------------


def test_min_blocks_threshold():
    """A request with cached_blocks < cache_affinity_min_blocks (default 2)
    should score 0 (cache-cold) and lose to the arrival-time order."""
    sched = create_cas_scheduler(cache_affinity_min_blocks=2)
    now = time.monotonic()
    req_a = make_request("a", arrival_time=now - 0.01)  # earlier, 1 cached block
    req_b = make_request("b", arrival_time=now - 0.00)  # later, 0 cached blocks

    for r in (req_a, req_b):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    # a has 1 block — below threshold of 2, so scored as 0
    _mock_get_computed_blocks(sched, {"a": block_size * 1, "b": 0})

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    # Both score 0; a has earlier arrival → a wins
    assert order == ["a", "b"]


# ---------------------------------------------------------------------------
# Test 5.5: explicit priority overrides cache affinity
# ---------------------------------------------------------------------------


def test_priority_overrides_cache():
    """A high-priority (low number) cache-cold request must beat a
    low-priority cache-warm request, regardless of cache score."""
    sched = create_cas_scheduler(policy="priority", cache_affinity_min_blocks=1)
    now = time.monotonic()
    req_a = make_request(
        "a", priority=10, arrival_time=now - 0.01
    )  # low priority, warm
    req_b = make_request(
        "b", priority=0, arrival_time=now - 0.00
    )  # high priority, cold

    for r in (req_a, req_b):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": block_size * 100, "b": 0})

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    assert order[0] == "b", f"High-priority request 'b' must be first; got {order}"
    assert order[1] == "a"


# ---------------------------------------------------------------------------
# Test 5.6: cache affinity breaks ties within the same priority class
# ---------------------------------------------------------------------------


def test_cache_tiebreak_within_priority():
    """Three requests with the same priority; the one with the most cached
    blocks should sort first within the priority class."""
    sched = create_cas_scheduler(
        policy="priority",
        cache_affinity_min_blocks=1,
        cache_affinity_bucket_edges=(4, 16, 64, 256),
    )
    now = time.monotonic()
    req_a = make_request("a", priority=5, arrival_time=now - 0.02)  # cold
    req_b = make_request(
        "b", priority=5, arrival_time=now - 0.01
    )  # 64 blocks — top bucket
    req_c = make_request(
        "c", priority=5, arrival_time=now - 0.00
    )  # 4 blocks — mid bucket

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    # b: 64 blocks → bucket 5 (≥ 256 edge), c: 4 blocks → bucket 2 ([4,16))
    _mock_get_computed_blocks(
        sched,
        {"a": 0, "b": block_size * 64, "c": block_size * 4},
    )

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    assert order[0] == "b", f"Expected 'b' first (most cached); got {order}"
    assert order[1] == "c", f"Expected 'c' second; got {order}"
    assert order[2] == "a", f"Expected 'a' last (cache-cold); got {order}"


# ---------------------------------------------------------------------------
# Test 5.7: starvation override promotes waiting request to absolute front
# ---------------------------------------------------------------------------


def test_starvation_override_promotes():
    """A request that has been waiting longer than cache_affinity_max_wait_s
    must sort before all non-starved requests, regardless of cache score."""
    max_wait = 0.2
    sched = create_cas_scheduler(
        cache_affinity_max_wait_s=max_wait,
        cache_affinity_min_blocks=1,
    )
    now = time.monotonic()
    # req_a is starved: waited 1 second
    req_a = make_request("a", arrival_time=now - 1.0)
    # req_b has lots of cache but is not starved
    req_b = make_request("b", arrival_time=now - 0.05)
    # req_c has some cache and is not starved
    req_c = make_request("c", arrival_time=now - 0.00)

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    _mock_get_computed_blocks(
        sched,
        {"a": 0, "b": block_size * 100, "c": block_size * 10},
    )

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    assert order[0] == "a", f"Starved request 'a' must be first; got {order}"
    # b has more cache than c → b before c
    assert order[1] == "b"
    assert order[2] == "c"


# ---------------------------------------------------------------------------
# Test 5.8: bucketing prevents thrash from small score differences
# ---------------------------------------------------------------------------


def test_bucketing_reduces_thrash():
    """Two requests with scores that fall in the same bucket should be tied on
    the cache axis and resolved purely by arrival_time."""
    sched = create_cas_scheduler(
        cache_affinity_min_blocks=1,
        cache_affinity_bucket_edges=(4, 16, 64, 256),
    )
    now = time.monotonic()
    # Both have scores 5 and 7, which are in the same bucket [4, 16)
    req_a = make_request("a", arrival_time=now - 0.01)  # earlier
    req_b = make_request("b", arrival_time=now - 0.00)  # later

    for r in (req_a, req_b):
        sched.waiting.add_request(r)

    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": block_size * 5, "b": block_size * 7})

    sched._reorder_waiting()

    order = _drain_sortable(sched)
    # Same bucket → tied on cache, arrival_time breaks tie → a first
    assert order == ["a", "b"], (
        f"Same-bucket requests should be ordered by arrival_time; got {order}"
    )


# ---------------------------------------------------------------------------
# Test 5.9: sticky-front preempted requests stay at the absolute front
# ---------------------------------------------------------------------------


def test_sticky_front_preserves_preempted_order():
    """Requests added via prepend_request() must remain at the absolute front
    in their prepend order, and must not be reordered by _reorder_waiting()."""
    sched = create_cas_scheduler(cache_affinity_min_blocks=1)
    now = time.monotonic()

    # Two "preempted" requests — added to sticky front
    req_p1 = make_request("p1", arrival_time=now - 0.5)
    req_p2 = make_request("p2", arrival_time=now - 0.4)
    # Three normal waiting requests with mixed scores
    req_n1 = make_request("n1", arrival_time=now - 0.03)  # warm
    req_n2 = make_request("n2", arrival_time=now - 0.02)  # cold
    req_n3 = make_request("n3", arrival_time=now - 0.01)  # warmer

    # Normal requests added via add_request (sortable)
    for r in (req_n1, req_n2, req_n3):
        sched.waiting.add_request(r)

    # Preempted requests added via prepend_request (sticky front)
    # prepend_request uses appendleft, so p2 added first → p1 ends up at front
    sched.waiting.prepend_request(req_p2)
    sched.waiting.prepend_request(req_p1)

    block_size = sched.block_size
    # n3: 20 blocks → bucket [16, 64) → index 3
    # n1:  5 blocks → bucket  [4, 16) → index 2
    # n2:  0 blocks → bucket 0 (cold)
    _mock_get_computed_blocks(
        sched,
        {"n1": block_size * 5, "n2": 0, "n3": block_size * 20, "p1": 0, "p2": 0},
    )

    sched._reorder_waiting()

    # Drain full queue: sticky first, then sortable
    order = _drain_queue(sched)

    # First two must be preempted (sticky front), in their prepend order
    assert order[:2] == ["p1", "p2"], (
        f"Preempted requests must be at front in prepend order; got {order[:2]}"
    )
    # Remaining sorted by cache: n3 (20 blocks, bucket 3) > n1 (5 blocks, bucket 2)
    #                            > n2 (0 blocks, bucket 0)
    assert order[2] == "n3"
    assert order[3] == "n1"
    assert order[4] == "n2"


# ---------------------------------------------------------------------------
# Test 5.10: disabled scheduler is a pass-through
# ---------------------------------------------------------------------------


def test_disabled_is_noop():
    """When cache_affinity_enabled=False, _reorder_waiting must not be invoked.
    Verified by the sort-latency counter: it must not grow after schedule()."""
    sched = create_cas_scheduler(cache_affinity_enabled=False)
    now = time.monotonic()
    req_a = make_request("a", arrival_time=now - 0.01)
    req_b = make_request("b", arrival_time=now - 0.00)
    req_c = make_request("c", arrival_time=now - 0.005)

    # Use add_request so requests are registered in self.requests (required by
    # schedule() → _update_after_schedule()).
    for r in (req_a, req_b, req_c):
        sched.add_request(r)

    before_samples = len(sched._stat_sort_us_samples)
    sched.schedule()
    # _reorder_waiting was not called (cache_affinity_enabled=False), so no new
    # sort-latency sample should have been recorded.
    assert len(sched._stat_sort_us_samples) == before_samples


# ---------------------------------------------------------------------------
# Test 5.11: KV manager exception is caught and treated as cache-cold
# ---------------------------------------------------------------------------


def test_kv_manager_exception_is_handled():
    """If kv_cache_manager.get_computed_blocks raises, the request should be
    treated as cache-cold (score 0) and schedule() must not propagate the
    exception."""
    sched = create_cas_scheduler(cache_affinity_min_blocks=1)
    now = time.monotonic()
    req_a = make_request("a", arrival_time=now - 0.01)  # will raise
    req_b = make_request("b", arrival_time=now - 0.00)  # returns 10 blocks

    for r in (req_a, req_b):
        sched.waiting.add_request(r)

    block_size = sched.block_size

    def _side_effect(req: Request):
        if req.request_id == "a":
            raise RuntimeError("simulated KV manager error")
        mock_blocks = MagicMock()
        mock_blocks.get_block_ids.return_value = (list(range(10)),)
        return mock_blocks, block_size * 10

    sched.kv_cache_manager.get_computed_blocks = Mock(side_effect=_side_effect)

    # Must not raise; req_a treated as cache-cold → stays behind req_b
    sched._reorder_waiting()

    order = _drain_sortable(sched)
    # b: 10 blocks (warm), a: 0 (exception → cold)
    assert order[0] == "b", f"Expected warm 'b' first; got {order}"
    assert order[1] == "a"


# ---------------------------------------------------------------------------
# Test 5.12: end-to-end plugin resolution via scheduler_cls
# ---------------------------------------------------------------------------


def test_resolves_via_scheduler_cls():
    """SchedulerConfig.get_scheduler_cls() must resolve the CacheAffinityScheduler
    class correctly when scheduler_cls is set as a dotted-path string."""
    from vllm.config import SchedulerConfig

    sched_cls_path = (
        "vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler"
    )
    cfg = SchedulerConfig.default_factory(scheduler_cls=sched_cls_path)
    resolved = cfg.get_scheduler_cls()

    assert resolved is CacheAffinityScheduler, (
        f"Expected CacheAffinityScheduler, got {resolved}"
    )
    assert issubclass(resolved, CacheAffinityScheduler)


# ---------------------------------------------------------------------------
# Test 5.13: sort-latency counter records non-zero samples
# ---------------------------------------------------------------------------


def test_sort_latency_recorded():
    """After _reorder_waiting() on a non-trivial queue, at least one sample
    must be recorded in _stat_sort_us_samples with a plausible value."""
    sched = create_cas_scheduler(cache_affinity_min_blocks=1)
    now = time.monotonic()

    # Add 50 requests
    for i in range(50):
        r = make_request(f"r{i:03d}", arrival_time=now - i * 0.001)
        sched.waiting.add_request(r)

    # Mock all to return 0 cached (simplest case; we just care about timing)
    mock_fn = Mock(return_value=(MagicMock(), 0))
    sched.kv_cache_manager.get_computed_blocks = mock_fn

    before = len(sched._stat_sort_us_samples)
    sched._reorder_waiting()
    after = len(sched._stat_sort_us_samples)

    assert after > before, "Expected at least one new sort-latency sample"

    last_sample = sched._stat_sort_us_samples[-1]
    assert last_sample >= 0, "Sort latency must be non-negative"
    assert last_sample < 100_000, (
        f"Sort latency {last_sample} µs exceeds 100 ms — suspiciously slow"
    )


# ---------------------------------------------------------------------------
# Test 5.14: batch-arrival guard triggers when spread is below threshold
# ---------------------------------------------------------------------------


def test_batch_guard_triggers_under_threshold():
    """All requests arriving simultaneously → reorder is skipped."""
    sched = create_cas_scheduler(
        cache_affinity_min_blocks=1,
        cache_affinity_batch_guard_threshold_s=0.01,
    )
    now = time.monotonic()
    # Three requests, all within 5 ms of each other.
    req_a = make_request("a", arrival_time=now + 0.000)
    req_b = make_request("b", arrival_time=now + 0.002)
    req_c = make_request("c", arrival_time=now + 0.004)

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    # b has lots of cached blocks — would normally be promoted, but the guard
    # should fire and leave the order unchanged.
    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": 0, "b": block_size * 64, "c": 0})

    before_guard = sched._stat_batch_guard_triggered_total
    sched._reorder_waiting()

    assert sched._stat_batch_guard_triggered_total == before_guard + 1, (
        "Guard counter should increment when spread < threshold"
    )
    # Order must be unchanged (guard skipped the sort).
    order = _drain_sortable(sched)
    assert order == ["a", "b", "c"], f"Guard should preserve FCFS order; got {order}"


# ---------------------------------------------------------------------------
# Test 5.15: batch-arrival guard is inactive when spread exceeds threshold
# ---------------------------------------------------------------------------


def test_batch_guard_skips_above_threshold():
    """Requests spread over > threshold → guard inactive, reorder happens."""
    sched = create_cas_scheduler(
        cache_affinity_min_blocks=1,
        cache_affinity_batch_guard_threshold_s=0.01,
    )
    now = time.monotonic()
    # Spread of 200 ms — well above 10 ms threshold.
    req_a = make_request("a", arrival_time=now + 0.000)
    req_b = make_request("b", arrival_time=now + 0.100)
    req_c = make_request("c", arrival_time=now + 0.200)

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    # b has the most cached blocks → should be promoted to front.
    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": 0, "b": block_size * 64, "c": 0})

    before_guard = sched._stat_batch_guard_triggered_total
    sched._reorder_waiting()

    assert sched._stat_batch_guard_triggered_total == before_guard, (
        "Guard counter must NOT increment when spread >= threshold"
    )
    order = _drain_sortable(sched)
    assert order[0] == "b", f"Cache-warm 'b' must be promoted; got {order}"


# ---------------------------------------------------------------------------
# Test 5.16: batch-arrival guard is disabled when threshold is zero
# ---------------------------------------------------------------------------


def test_batch_guard_threshold_zero_disables():
    """threshold=0 disables the guard entirely; reorder happens even with
    identical arrival times."""
    sched = create_cas_scheduler(
        cache_affinity_min_blocks=1,
        cache_affinity_batch_guard_threshold_s=0.0,
    )
    now = time.monotonic()
    # All three requests arrive at exactly the same instant.
    req_a = make_request("a", arrival_time=now)
    req_b = make_request("b", arrival_time=now)
    req_c = make_request("c", arrival_time=now)

    for r in (req_a, req_b, req_c):
        sched.waiting.add_request(r)

    # b is cache-warm → should be promoted despite zero spread.
    block_size = sched.block_size
    _mock_get_computed_blocks(sched, {"a": 0, "b": block_size * 64, "c": 0})

    sched._reorder_waiting()

    assert sched._stat_batch_guard_triggered_total == 0, (
        "Guard must never trigger when threshold=0"
    )
    order = _drain_sortable(sched)
    assert order[0] == "b", (
        f"With guard disabled, cache-warm 'b' must still be promoted; got {order}"
    )
