# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-affinity-aware scheduler plugin for vLLM V1.

Reorders the waiting queue by cached-prefix length before each
scheduling iteration. This is the in-engine equivalent of sglang's
RadixAttention scheduling, implemented on top of vLLM's existing
block-hash prefix cache (no token-level radix tree, no KV-manager
changes, no request-schema changes).

Load with::

    --scheduler-cls vllm.v1.core.sched.cache_affinity_scheduler.CacheAffinityScheduler

Composes with priority scheduling: cache affinity is a tiebreaker
within a priority class, never across classes. A high-priority
cache-cold request always beats a low-priority cache-warm request.

Anti-starvation: a request waiting longer than
``cache_affinity_max_wait_s`` seconds is promoted to the head
regardless of cache score.

Mutually exclusive at engine-load time with EWSJF (PR #33392) and
the default FCFS/priority schedulers — pick one via ``--scheduler-cls``.
"""

import time
from collections import deque
from collections.abc import Callable, Iterator

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)


class CacheAffinityRequestQueue(RequestQueue):
    """Deque-backed waiting queue with per-iteration cache-affinity re-sort.

    Maintains two internal collections:
    - ``_sticky``: requests added via ``prepend_request`` (preempted requests
      that must remain at the front and are not reordered by cache affinity).
    - ``_sortable``: requests added via ``add_request`` (the bulk of waiting
      requests that are candidates for cache-affinity reordering).

    ``pop_request`` drains ``_sticky`` first, then ``_sortable``.
    ``resort`` only sorts ``_sortable``; ``_sticky`` is left untouched.
    """

    def __init__(self) -> None:
        self._sticky: deque[Request] = deque()
        self._sortable: deque[Request] = deque()

    # ------------------------------------------------------------------
    # RequestQueue ABC implementation
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self._sortable.append(request)

    def pop_request(self) -> Request:
        if self._sticky:
            return self._sticky.popleft()
        return self._sortable.popleft()

    def peek_request(self) -> Request:
        if self._sticky:
            return self._sticky[0]
        return self._sortable[0]

    def prepend_request(self, request: Request) -> None:
        """Add request to the sticky front (preserves preemption order)."""
        self._sticky.appendleft(request)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests to the sticky front (preserves reversed order,
        matching FCFSRequestQueue.extendleft behaviour)."""
        for req in requests:
            self._sticky.appendleft(req)

    def remove_request(self, request: Request) -> None:
        try:
            self._sticky.remove(request)
        except ValueError:
            self._sortable.remove(request)

    def remove_requests(self, requests: "RequestQueue | list[Request]") -> None:  # type: ignore[override]
        req_set = set(requests)
        self._sticky = deque(r for r in self._sticky if r not in req_set)
        self._sortable = deque(r for r in self._sortable if r not in req_set)

    def __bool__(self) -> bool:
        return bool(self._sticky) or bool(self._sortable)

    def __len__(self) -> int:
        return len(self._sticky) + len(self._sortable)

    def __iter__(self) -> Iterator[Request]:
        """Iterate sticky-front first, then sortable (policy order)."""
        yield from self._sticky
        yield from self._sortable

    # ------------------------------------------------------------------
    # Extension: cache-affinity re-sort
    # ------------------------------------------------------------------

    def iter_sortable(self) -> Iterator[Request]:
        """Iterate only the sortable portion (excludes sticky-front)."""
        return iter(self._sortable)

    def resort(self, key_fn: Callable[[Request], tuple]) -> None:
        """Re-sort the sortable deque in-place by ``key_fn`` (lower = better).

        The sticky-front deque is left untouched so preempted requests
        remain at the head in their original preemption order.
        """
        self._sortable = deque(sorted(self._sortable, key=key_fn))


class CacheAffinityScheduler(Scheduler):
    """Scheduler subclass that reorders the waiting queue by cached-prefix
    length before each scheduling iteration.

    Inherits all scheduling logic from ``Scheduler``; only the waiting-queue
    ordering is changed.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)

        cfg = self.scheduler_config
        self.cache_affinity_enabled: bool = getattr(cfg, "cache_affinity_enabled", True)
        self.cache_affinity_max_wait_s: float = getattr(
            cfg, "cache_affinity_max_wait_s", 0.05
        )
        self.cache_affinity_min_blocks: int = getattr(
            cfg, "cache_affinity_min_blocks", 2
        )
        self.cache_affinity_bucket_edges: tuple[int, ...] = tuple(
            sorted(getattr(cfg, "cache_affinity_bucket_edges", (4, 16, 64, 256)))
        )
        self.cache_affinity_batch_guard_threshold_s: float = getattr(
            cfg, "cache_affinity_batch_guard_threshold_s", 0.01
        )

        # Replace self.waiting with our queue, migrating any existing contents.
        # In practice self.waiting will be empty at construction time, but
        # defend against edge cases. Use an explicitly typed intermediate so
        # mypy (running with --follow-imports skip) can resolve the attribute.
        new_waiting: CacheAffinityRequestQueue = CacheAffinityRequestQueue()
        old_waiting: RequestQueue = self.waiting  # type: ignore[has-type]
        for req in old_waiting:
            new_waiting.add_request(req)
        self.waiting = new_waiting

        # Per-iteration scratch: block IDs considered cached during the most
        # recent reorder, keyed by request_id. Used to detect thrash (blocks
        # evicted between scoring and admission). Populated in _reorder_waiting;
        # consumed in _record_thrash_metric.
        self._last_iter_cached_blocks: dict[str, set[int]] = {}

        # Local observable counters (Prometheus wiring is a follow-up task).
        self._stat_thrash_evictions_total: int = 0
        self._stat_batch_guard_triggered_total: int = 0
        self._stat_sort_us_samples: list[int] = []
        _SORT_SAMPLE_CAP = 1000  # avoid unbounded growth
        self._sort_sample_cap = _SORT_SAMPLE_CAP

        logger.info_once(
            "CacheAffinityScheduler enabled. enabled=%s max_wait_s=%s "
            "min_blocks=%s bucket_edges=%s batch_guard_threshold_s=%s",
            self.cache_affinity_enabled,
            self.cache_affinity_max_wait_s,
            self.cache_affinity_min_blocks,
            self.cache_affinity_bucket_edges,
            self.cache_affinity_batch_guard_threshold_s,
        )

    # ------------------------------------------------------------------
    # Public interface (overrides Scheduler.schedule)
    # ------------------------------------------------------------------

    def schedule(self):  # type: ignore[override]
        if self.cache_affinity_enabled and len(self.waiting) > 1:
            self._reorder_waiting()
        output = super().schedule()
        self._record_thrash_metric(output)
        return output

    # ------------------------------------------------------------------
    # Core reordering logic
    # ------------------------------------------------------------------

    def _reorder_waiting(self) -> None:
        """Score and re-sort the sortable portion of the waiting queue."""
        sort_start = time.monotonic_ns()
        now = time.monotonic()

        requests_to_score = list(self.waiting.iter_sortable())  # type: ignore[attr-defined]
        if len(requests_to_score) <= 1:
            return

        # Batch-arrival guard: if all waiting requests arrived within the
        # guard threshold, FCFS order is already optimal for cache utilization
        # (sequential same-prefix requests are already adjacent). Skip the
        # reorder to avoid pure overhead.
        if self.cache_affinity_batch_guard_threshold_s > 0:
            arrival_times = [r.arrival_time for r in requests_to_score]
            if (
                max(arrival_times) - min(arrival_times)
                < self.cache_affinity_batch_guard_threshold_s
            ):
                self._stat_batch_guard_triggered_total += 1
                return

        scored: dict[str, int] = {}
        starved: set[str] = set()
        cached_blocks_per_req: dict[str, set[int]] = {}

        for req in requests_to_score:
            wait_s = now - req.arrival_time
            if wait_s > self.cache_affinity_max_wait_s:
                starved.add(req.request_id)
                scored[req.request_id] = -1  # sentinel; sort key handles separately
                continue

            if req.num_computed_tokens > 0:
                # KVTransfer: request already has some tokens computed.
                # Use existing count instead of re-querying.
                num_cached_blocks = req.num_computed_tokens // self.block_size
            else:
                try:
                    blocks_obj, num_cached_tokens = (
                        self.kv_cache_manager.get_computed_blocks(req)
                    )
                except Exception:
                    # KV manager call should not fail in normal operation.
                    # If it does, treat as cache-cold and continue — never
                    # let scoring crash schedule().
                    scored[req.request_id] = 0
                    continue

                num_cached_blocks = num_cached_tokens // self.block_size

                # Stash block IDs for the thrash metric (best-effort).
                try:
                    block_ids_nested = blocks_obj.get_block_ids(allow_none=True)
                    if block_ids_nested is not None:
                        cached_blocks_per_req[req.request_id] = {
                            bid for group in block_ids_nested for bid in group
                        }
                except Exception:
                    pass  # Thrash tracking is optional; never crash schedule()

            if num_cached_blocks < self.cache_affinity_min_blocks:
                num_cached_blocks = 0
            scored[req.request_id] = num_cached_blocks

        self._last_iter_cached_blocks = cached_blocks_per_req

        bucketed = {rid: self._bucket(s) for rid, s in scored.items()}
        policy = self.scheduler_config.policy

        def sort_key(req: Request) -> tuple:
            if req.request_id in starved:
                # Absolute front: sorts before any valid priority (>=0) or
                # FCFS leading-zero key.
                return (-1, 0, 0.0, req.request_id)
            b = bucketed[req.request_id]
            if policy == "priority":
                # Primary: priority (lower = higher priority).
                # Secondary: -bucket (higher cache affinity = better).
                # Tertiary: arrival_time (earlier = better).
                return (req.priority, -b, req.arrival_time, req.request_id)
            else:
                # FCFS: sort by -bucket then arrival_time; leading 0 keeps
                # tuple length uniform with the priority branch.
                return (0, -b, req.arrival_time, req.request_id)

        self.waiting.resort(sort_key)  # type: ignore[attr-defined]

        sort_us = (time.monotonic_ns() - sort_start) // 1000
        self._observe_sort_latency(int(sort_us))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bucket(self, score: int) -> int:
        """Map a raw cached-block count to a bucket index (0 = cache-cold).

        Requests in the same bucket are tied on the cache axis; ties are
        broken by arrival_time. This reduces sort thrash across iterations
        when scores drift by a small amount.
        """
        if score <= 0:
            return 0
        for i, edge in enumerate(self.cache_affinity_bucket_edges):
            if score < edge:
                return i + 1
        return len(self.cache_affinity_bucket_edges) + 1

    def _observe_sort_latency(self, sort_us: int) -> None:
        """Record a sort-latency sample for the current iteration.

        Capped at ``_sort_sample_cap`` entries to prevent unbounded growth.
        Prometheus wiring is deferred to a follow-up task.
        """
        if len(self._stat_sort_us_samples) >= self._sort_sample_cap:
            self._stat_sort_us_samples.pop(0)
        self._stat_sort_us_samples.append(sort_us)

    def _record_thrash_metric(self, output) -> None:
        """Detect blocks that were considered cached during scoring but were
        evicted before the request was admitted.

        Full thrash detection requires comparing pre-scoring block IDs against
        the actually-allocated blocks returned by allocate_slots, which is not
        directly exposed by SchedulerOutput. Deferring the per-block comparison
        to a follow-up task; for now only the counter is wired up.

        TODO: wire fine-grained thrash detection once SchedulerOutput exposes
        per-request cached-block IDs before vs. after allocation.
        """
        pass
