"""Exact C cache methods with bounded CPU index/allocator stubs; no GPU proof."""
from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import numpy as np

from test_latent_ipc_patch import PinnedCase


class Indices(list):
    """Index-only stub, NOT a Torch tensor or mathematical reference."""
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Indices(result) if isinstance(key, slice) else result
    def clone(self): return Indices(self)
    def to(self, dtype=None, copy=False): return Indices(self)
    def numel(self): return len(self)


class Tensor(np.ndarray):
    """numpy stand-in for the few Torch tensor methods the OCC-3 guards call.

    It keeps dtype, layout and a device label, so it can show the int32 row vs
    int64 prefix contract. It is NOT a CUDA or Triton model.
    """
    def __array_finalize__(self, obj):
        self._device = getattr(obj, "_device", "cuda:0")
    @property
    def device(self): return self._device  # numpy 2 ndarray.device is read-only
    def _keep(self, out):
        out = out.view(Tensor)
        out._device = self._device
        return out
    def to(self, dtype=None, copy=False): return self._keep(np.array(self, dtype=dtype, copy=True))
    def clone(self): return self._keep(np.array(self, copy=True))
    def dim(self): return self.ndim
    def is_contiguous(self): return bool(self.flags["C_CONTIGUOUS"])


def tensor(values, dtype=np.int64, device="cuda:0"):
    out = np.array(values, dtype=dtype).view(Tensor)
    out._device = device
    return out


NP_TORCH = NS(int64=np.int64, int32=np.int32,
              arange=lambda n, device="cuda:0": tensor(np.arange(n), device=device),
              sort=lambda values: NS(values=np.sort(values)),
              stack=lambda values: np.stack(values))


class Log:
    def __init__(self): self.records = []
    def warning(self, *args): self.records.append(("warning",) + args)
    def error(self, *args): self.records.append(("error",) + args)


class RowPool:
    def __init__(self, row):
        self.row, self.frees = Indices(row), 0
        self.req_to_token = self
    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, selection = key
            if index != 3: raise AssertionError("wrong request row")
            return self.row[selection]
        return self.row
    def free(self, req):
        if req.kv.req_pool_idx is None: raise AssertionError("double row free")
        self.frees += 1
        req.kv.req_pool_idx = None


class Kv:
    def __init__(self, committed=0, allocated=0, index=3):
        self.req_pool_idx, self.kv_committed_len, self.kv_allocated_len = index, committed, allocated
        self.cache_protected_len, self.max_evicted_seqlen, self.holds_mamba = 0, 0, False
    @property
    def holds_kv(self): return self.req_pool_idx is not None
    @property
    def is_kv_released(self): return self.kv_allocated_len == 0
    def mark_kv_released(self): self.kv_allocated_len = 0


class Cache:
    def __init__(self, row=range(64), page=4):
        self.req_to_token_pool = RowPool(row)
        self.root_node = object()
        self.calls, self.freed_pages, self.page = [], set(), page
    def root_node_handle(self, extra_key=None):
        return self.root_node
    def free_kv_row(self, kv, ranges):
        self.calls.append(("free", ranges))
        for start, end in ranges:
            for page in {self.req_to_token_pool.row[i] // self.page for i in range(start, end)}:
                if page in self.freed_pages: raise AssertionError("double page free")
                self.freed_pages.add(page)
    def __getattr__(self, name):
        if name in {"match_prefix", "insert", "insert_req", "cache_unfinished_req", "claim_kv_row", "unpin", "on_release", "backup_kv_cache"}:
            def forbidden(*args, **kwargs): raise AssertionError("private cache contamination: " + name)
            return forbidden
        raise AttributeError(name)


def request(committed=5, allocated=8, end=5):
    return NS(occ3_private_prefix=True, skip_radix_cache_insert=True, kv=Kv(committed, allocated),
              prefix_indices=Indices(), extend_range=NS(end=end), session=None, extra_key=None,
              last_node=None, host_hit_length=0, storage_hit_length=0, occ3_kv_released=False,
              occ3_scheduler=NS(tp_group=NS(ranks=[0])), occ3_kv_quarantined=False, occ3_kv_release_started=False)


def page_row(pages, allocated, page=64):
    """Page-exact slots: slot i is offset i % page of page pages[i // page]."""
    return np.concatenate([p * page + np.arange(page) for p in pages])[:allocated]


def paged_cache(row, page=64, pages=16):
    """Cache whose request row 3 is an int32 table row, like ReqToTokenPool.

    The allocator stub has the stock paged size, so valid pages are [1, pages].
    """
    tree = Cache(row=[int(slot) for slot in row], page=page)
    table = np.zeros((4, len(row)), dtype=np.int32)
    table[3] = row
    tree.req_to_token_pool.req_to_token = tensor(table, np.int32)
    tree.token_to_kv_pool_allocator = NS(page_size=page, size=page * pages)
    return tree


class TestPrivateCache(PinnedCase):
    def setUp(self):
        # The external fence is an explicit boundary stub in this allocator suite.
        # Actual stream/TP ownership is covered only by root integration gates.
        self.fence_calls=[]
        self.worker=types.ModuleType("occ3_sglang.latent_worker")
        self.worker.fence_private_kv_release=lambda scheduler,req:self.fence_calls.append(req)
        self.worker_patch=patch.dict("sys.modules",{self.worker.__name__:self.worker})
        self.worker_patch.start()
        self.addCleanup(self.worker_patch.stop)

    def functions(self, torch_module=NP_TORCH):
        path = "mem_cache/common.py"
        match = self.sources.function(path, "occ3_private_match", MatchResult=lambda **kwargs: NS(**kwargs))
        # The page check is stubbed here; the list-based rows below include page 0.
        # test_private_release_* exercise the real check on numpy rows.
        release = self.sources.function(path, "occ3_release_private_kv",
                                        occ3_private_release_fault=lambda *args: None)
        cache = self.sources.function(path, "maybe_cache_unfinished_req", occ3_private_match=match,
                                      torch=torch_module)
        public_release = self.sources.function(path, "release_kv_cache", occ3_release_private_kv=release)
        return match, release, cache, public_release

    def test_private_match_native_root_handles_and_namespace(self):
        """Pinned Python/Rust adapters; no model or GPU allocation."""
        import hashlib
        from _source_pins import pristine_root, srt_dir
        source = srt_dir(pristine_root())
        pins = {'mem_cache/base_prefix_cache.py': '49f20555e6b96af071db38a1bd9829b322ce76381940bd2f340abced151335a9', 'mem_cache/unified_radix_cache.py': 'c6f1764166c371cf050f05c966daed8ded366365163a69f066569cc74e2575c3', 'mem_cache/rust_tree_core/adapter.py': '25c567c21768b2b5c23b4a9f63705f6cbde73b98b9a2f65323c088bd2deef3d2'}
        for path, expected in pins.items():
            if not (source / path).is_file():
                self.skipTest("Pinned SGLang source absent; set OCC3_SGLANG_SOURCE_BASE")
            text = (source / path).read_text(encoding="utf-8")
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected)
            self.sources.text[path] = text
        fn = self.sources.function
        base = fn("mem_cache/base_prefix_cache.py", "root_node_handle", "BasePrefixCache")
        unified = "mem_cache/unified_radix_cache.py"
        rust = "mem_cache/rust_tree_core/adapter.py"
        RustCore = type("PinnedRustCoreMethods", (), {
            "root_node": property(fn(rust, "root_node", "RustUnifiedTreeCore")),
            "root_node_handle": fn(rust, "root_node_handle", "RustUnifiedTreeCore"),
        })
        UnifiedCache = type("PinnedUnifiedCacheMethods", (), {
            "root_node": property(fn(unified, "root_node", "UnifiedRadixCache")),
            "root_node_handle": fn(unified, "root_node_handle", "UnifiedRadixCache"),
        })
        keys = []
        def root_for(key):
            keys.append(key)
            return {"private-a": 41, "private-b": 57}[key]
        core = RustCore()
        core._binding = NS(root_node_handle=root_for)
        tree = UnifiedCache()
        tree.tree_core = core
        legacy = type("PinnedLegacyRootMethod", (), {"root_node_handle": base})()
        legacy.root_node = object()
        match, _, _, _ = self.functions()
        for cache, key, expected in ((tree, "private-a", 41), (tree, "private-b", 57), (legacy, None, legacy.root_node)):
            prefix = object()
            req = NS(skip_radix_cache_insert=True, kv=NS(cache_protected_len=0), prefix_indices=prefix, extra_key=key)
            hit = match(req, cache)
            self.assertIs(hit.device_indices, prefix)
            self.assertEqual((hit.last_device_node, hit.last_host_node, hit.best_match_node), (expected,) * 3)
            self.assertEqual((req.last_node, req.last_host_node, req.best_match_node), (expected,) * 3)
            self.assertEqual(hit.cache_protected_len, 0)
            self.assertEqual(req.num_matched_prefix_tokens, 0)
        self.assertEqual(keys, ["private-a", "private-b"])

    def test_private_chunk_progress_keeps_partial_page_and_not_overshoot(self):
        match, _, cache, _ = self.functions()
        tree, req = Cache(), request()
        cache(req, tree, chunked=True)
        self.assertEqual(req.prefix_indices, [0, 1, 2, 3, 4])
        self.assertEqual(req.kv.cache_protected_len, 0)
        req.kv.kv_committed_len, req.kv.kv_allocated_len, req.extend_range.end = 11, 16, 9
        cache(req, tree, chunked=True)
        self.assertEqual(req.prefix_indices, list(range(9)))
        tree.req_to_token_pool.row[0] = 999
        self.assertEqual(req.prefix_indices[0], 0, "private indices must not alias mutable row")
        hit = match(req, tree)
        self.assertEqual(hit.device_indices, list(range(9)))
        self.assertEqual(hit.cache_protected_len, 0)
        self.assertIs(req.last_node, tree.root_node)
        self.assertEqual(req.num_matched_prefix_tokens, 0)

    def test_private_ownership_rejects_protected_or_regressing_prefix(self):
        _, _, cache, _ = self.functions()
        for protected, old_prefix, committed, allocated in [(1, 0, 5, 8), (0, 6, 5, 8), (0, 0, 9, 4)]:
            req, tree = request(committed, allocated, end=9), Cache()
            req.kv.cache_protected_len = protected
            req.prefix_indices = Indices(range(old_prefix))
            with self.assertRaisesRegex(RuntimeError, "ownership"): cache(req, tree)
            self.assertEqual(tree.calls, [])

    def test_partial_page_and_overshoot_free_exactly_once(self):
        _, _, _, release = self.functions()
        for page in (1, 4, 16):
            for committed, allocated in [(1, 1), (3, 9), (4, 8), (5, 16), (16, 17), (0, 6)]:
                with self.subTest(page=page, committed=committed, allocated=allocated):
                    req, tree = request(committed, allocated), Cache(page=page)
                    release(req, tree)
                    release(req, tree)
                    self.assertEqual(tree.calls, [("free", [(0, allocated)])])
                    self.assertEqual(tree.req_to_token_pool.frees, 1)
                    self.assertEqual(tree.freed_pages, set(range((allocated + page - 1) // page)))
                    self.assertFalse(req.kv.holds_kv)
                    self.assertEqual(req.kv.kv_allocated_len, 0)
                    self.assertEqual(req.kv.kv_committed_len, 0)

    def test_skip_flag_alone_preserves_original_behavior(self):
        _, _, cache, _ = self.functions()
        req = NS(occ3_private_prefix=False, skip_radix_cache_insert=True)
        tree = Cache()
        cache(req, tree)
        self.assertEqual(tree.calls, [])
        req.skip_radix_cache_insert = False
        tree.cache_unfinished_req = lambda req, **kw: tree.calls.append(("ordinary", kw))
        cache(req, tree, chunked=True)
        self.assertEqual(tree.calls, [("ordinary", {"chunked": True})])

    def test_policy_private_bypasses_both_trees_and_retains_indices(self):
        match, _, _, _ = self.functions()
        common = types.ModuleType("sglang.srt.mem_cache.common")
        common.occ3_private_match = match
        fn = self.sources.function("managers/schedule_policy.py", "match_prefix_for_req")
        scan = self.sources.function("managers/schedule_policy.py", "_compute_prefix_matches", "SchedulePolicy", match_prefix_for_req=fn)
        tree = Cache()
        temporary = NS(reset=lambda: None, match_prefix=lambda *a: self.fail("temporary match"), insert=lambda *a: self.fail("temporary insert"))
        req = request(); req.prefix_indices = Indices(range(5)); req.rid = "private"
        with patch.dict("sys.modules", {common.__name__: common}):
            hit = fn(tree, req)
            self.assertEqual(hit.device_indices, list(range(5)))
            result = scan(NS(tree_cache=tree, waiting_queue_radix_tree=temporary), [req], None)
        self.assertEqual(result, set())
        self.assertEqual(req.prefix_indices, list(range(5)))

    def test_req_init_next_round_no_tree_match_or_host_reentry(self):
        match, _, _, _ = self.functions()
        common = types.ModuleType("sglang.srt.mem_cache.common"); common.occ3_private_match = match
        fn = self.sources.function("managers/schedule_batch.py", "init_next_round_input", "Req")
        req = request(); req.is_dllm = lambda: False; req.is_retracted = False
        req.full_untruncated_fill_ids = list(range(20)); req._refresh_fill_ids = lambda: None
        req.prefix_indices = Indices(range(5))
        with patch.dict("sys.modules", {common.__name__: common}):
            fn(req, Cache())
            fn(req, None)
            self.assertEqual(req.prefix_indices, list(range(5)))
            req.is_retracted = True
            with self.assertRaisesRegex(RuntimeError, "retraction_unsupported"): fn(req, Cache())

    def test_no_private_backup_or_offload(self):
        fn = self.sources.function("mem_cache/common.py", "backup_kv_cache")
        req = request()
        self.assertFalse(fn(req, Cache(), None, None, "cpu_tensor"))
        self.assertFalse(fn(req, Cache(), None, None, "host_pool"))
        offload = self.sources.function("managers/schedule_batch.py", "offload_kv_cache", "Req")
        with self.assertRaisesRegex(RuntimeError, "export_forbidden"): offload(req, None, None)

    def test_retraction_fences_before_free_no_reset_replay(self):
        worker = types.ModuleType("occ3_sglang.latent_worker")
        events = []
        worker.abort_request = lambda *a: events.append("fence")
        fn = self.sources.function("managers/schedule_batch.py", "release_req",
            release_kv_cache=lambda *a, **kw: events.append("free"),
            FINISH_ABORT=lambda *a: NS(message=a[0]), HTTPStatus=NS(SERVICE_UNAVAILABLE=503))
        req = request(); req.occ3_force_eager = True; req.occ3_scheduler = object()
        req.reset_for_retract = lambda: self.fail("retraction replay reset")
        with patch.dict("sys.modules", {worker.__name__: worker}):
            self.assertFalse(fn(req=req, remaing_req_count=0, req_to_token_pool=None,
                                token_to_kv_pool_allocator=None, tree_cache=None, hisparse_coordinator=None))
        self.assertEqual(events, ["fence", "free"])
        self.assertEqual(req.to_finish.message, "occ3_retraction_unsupported")

    @unittest.skipUnless(os.environ.get("OCC3_ROOT_IMAGE_CPU_TESTS") == "1", "root-only real Torch CPU index ownership test")
    def test_root_only_actual_torch_index_clone(self):
        import torch
        match, _, cache, _ = self.functions(torch_module=torch)
        req, tree = request(), Cache()
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        # The real ReqToTokenPool row is int32; the stash must widen it.
        tree.req_to_token_pool.req_to_token = torch.arange(64, dtype=torch.int32).repeat(4, 1)
        cache(req, tree)
        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        self.assertEqual(req.prefix_indices.tolist(), list(range(5)))
        tree.req_to_token_pool.req_to_token[3, 0] = 999
        self.assertEqual(req.prefix_indices[0].item(), 0)

    def test_failed_fence_quarantines_without_pool_reuse(self):
        _,release,_,_=self.functions()
        req,tree=request(),Cache()
        def fail(scheduler,req):raise RuntimeError("uncertain CUDA fence")
        self.worker.fence_private_kv_release=fail
        with self.assertRaisesRegex(RuntimeError,"quarantined"):release(req,tree)
        self.assertTrue(req.occ3_kv_quarantined)
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(tree.calls,[])
        self.assertEqual(tree.req_to_token_pool.frees,0)
        # A later caller may not retry a quarantined row into the ordinary pool.
        self.worker.fence_private_kv_release=lambda *a:None
        with self.assertRaisesRegex(RuntimeError,"quarantined"):release(req,tree)
        self.assertEqual(tree.calls,[])

    def test_partial_release_failure_cannot_doublefree_pages(self):
        _,release,_,_=self.functions()
        req,tree=request(),Cache()
        def fail(req):raise RuntimeError("row release failed after page release")
        tree.req_to_token_pool.free=fail
        with self.assertRaisesRegex(RuntimeError,"quarantined"):release(req,tree)
        self.assertTrue(req.occ3_kv_quarantined)
        self.assertTrue(req.occ3_kv_release_started)
        self.assertEqual(len(tree.calls),1)
        self.assertEqual(len(self.fence_calls),1)
        with self.assertRaisesRegex(RuntimeError,"quarantined"):release(req,tree)
        self.assertEqual(len(tree.calls),1)

    def test_private_stash_widens_int32_row_to_independent_int64_prefix(self):
        _, _, cache, _ = self.functions()
        req, tree = request(), Cache()
        req.prefix_indices = tensor([])
        table = tensor(np.arange(256).reshape(4, 64), np.int32)
        tree.req_to_token_pool.req_to_token = table
        cache(req, tree)
        self.assertEqual(req.prefix_indices.dtype, np.int64)
        # alloc_for_extend's writer loads prefix_indices through an int64 pointer.
        as_int64 = np.frombuffer(req.prefix_indices.tobytes(), dtype=np.int64)
        self.assertEqual(as_int64.tolist(), list(range(192, 197)))
        req.prefix_indices[0] = -1
        self.assertEqual(int(table[3, 0]), 192)

    def test_private_prefix_fault_names_writer_abi_mismatches(self):
        fault = self.sources.function("mem_cache/common.py", "occ3_private_prefix_fault", torch=NP_TORCH)
        table = tensor(np.zeros((4, 64)), np.int32)
        def req(prefix, committed=8, allocated=8, holds=True):
            return NS(prefix_indices=prefix,
                      kv=NS(holds_kv=holds, kv_committed_len=committed, kv_allocated_len=allocated))
        good = tensor(range(8))
        self.assertIsNone(fault(req(good), table))
        self.assertIsNone(fault(req(tensor([], np.int32)), table))
        cases = [("dtype", req(tensor(range(8), np.int32))),
                 ("layout", req(tensor(range(16))[::2])),
                 ("layout", req(tensor(np.arange(8).reshape(2, 4)))),
                 ("device", req(tensor(range(8), device="cpu"))),
                 ("length", req(tensor(range(9)))),
                 ("length", req(good, holds=False)),
                 ("length", req(good, allocated=6))]
        for name, bad in cases:
            with self.subTest(name):
                self.assertEqual(fault(bad, table), name)

    def test_bad_private_chunk_prefix_is_declined_before_the_writer(self):
        log, aborted = Log(), []
        common = types.ModuleType("sglang.srt.mem_cache.common")
        common.occ3_private_prefix_fault = self.sources.function(
            "mem_cache/common.py", "occ3_private_prefix_fault", torch=NP_TORCH)
        decline = self.sources.function("managers/scheduler.py", "_occ3_decline_bad_private_prefix",
                                        "Scheduler", logger=log)
        table = tensor(np.zeros((4, 64)), np.int32)
        def scheduler(prefix, private=True):
            chunk = NS(rid="r1", occ3_private_prefix=private, prefix_indices=prefix,
                       kv=NS(holds_kv=True, kv_committed_len=8, kv_allocated_len=8))
            return NS(chunked_req=chunk, tree_cache=NS(req_to_token_pool=NS(req_to_token=table)),
                      _occ3_abort_one=lambda req, reason: aborted.append((req, reason)))
        with patch.dict("sys.modules", {common.__name__: common}):
            for ok in (scheduler(tensor(range(8))), scheduler(tensor(range(8), np.int32), private=False),
                       NS(chunked_req=None)):
                decline(ok)
            self.assertEqual(aborted, [])
            bad = scheduler(tensor(range(8), np.int32))  # the pre-fix int32 copy
            decline(bad)
        self.assertEqual(aborted, [(bad.chunked_req, "private_prefix_invalid")])
        self.assertEqual([record[0] for record in log.records], ["error"])
        # The decline runs after the chunk stash and before the chunk's next round,
        # so the raw-pointer writer in prepare_for_extend never sees a bad prefix.
        text = self.sources.text["managers/scheduler.py"]
        start = text.index("    def _get_new_batch_prefill_raw(")
        body = text[start:text.index("\n    def ", start + 1)]
        guard = body.index("self._occ3_decline_bad_private_prefix()")
        self.assertLess(guard, body.index("self.chunked_req.init_next_round_input()"))
        self.assertLess(guard, body.index("add_chunked_req("))
        self.assertLess(guard, body.index("prepare_for_extend("))

    def test_private_row_fault_requires_page_exact_unique_in_range_rows(self):
        fault = self.sources.function("mem_cache/common.py", "occ3_private_row_fault", torch=NP_TORCH)
        def check(row):
            return fault(paged_cache(row), Kv(index=3), len(row))
        good = page_row([7, 3, 12], 138)  # nonconsecutive pages, partial last page
        self.assertIsNone(check(good))
        self.assertIsNone(check(page_row([16], 1)))
        shifted = good.copy(); shifted[64 + 5] += 1
        straddle = good.copy(); straddle[96:128] = 4 * 64 + np.arange(32, 64)
        # Pre-fix shape: an int32 row loaded through an int64 pointer and stored
        # back as int32 keeps every other slot.
        half = good.astype(np.int32).view(np.int64).astype(np.int32)
        scrambled = np.concatenate([half, half])
        cases = [("layout", shifted), ("layout", straddle), ("layout", scrambled),
                 ("range", page_row([0, 3], 100)), ("range", page_row([7, 17], 100)),
                 ("duplicate", page_row([7, 3, 7], 138))]
        for name, row in cases:
            with self.subTest(name):
                self.assertEqual(check(row), name)

    def quarantine_owner(self, tree, log, ranks=(0,), group=None, capacity=4096):
        """Scheduler stand-in carrying the manifest-applied quarantine method."""
        owner = NS(token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator,
                   max_total_num_tokens=capacity, tp_group=NS(ranks=list(ranks)),
                   tp_cpu_group=group, occ3_quarantined_tokens=0,
                   _occ3_quarantine_error_logged=False)
        method = self.sources.function("managers/scheduler.py", "_occ3_quarantine_private_kv",
                                       "Scheduler", logger=log)
        owner._occ3_quarantine_private_kv = types.MethodType(method, owner)
        return owner

    def test_private_release_leaks_untrusted_row_and_frees_exact_pages(self):
        log = Log()
        fault = self.sources.function("mem_cache/common.py", "occ3_private_row_fault", torch=NP_TORCH)
        decision = self.sources.function("mem_cache/common.py", "occ3_private_release_fault",
                                         occ3_private_row_fault=fault, torch=NP_TORCH)
        release = self.sources.function("mem_cache/common.py", "occ3_release_private_kv",
                                        occ3_private_release_fault=decision, logger=log)
        req, tree = request(138, 138), paged_cache(page_row([7, 3, 12], 138))
        release(req, tree)
        self.assertEqual(tree.calls, [("free", [(0, 138)])])
        self.assertEqual(tree.freed_pages, {7, 3, 12})
        self.assertTrue(req.occ3_kv_released)
        self.assertEqual(log.records, [])
        totals = []
        for row in (page_row([7, 3, 7], 138), page_row([0, 3], 100)):
            req, tree = request(len(row), len(row)), paged_cache(row)
            req.occ3_scheduler = owner = self.quarantine_owner(tree, log)
            release(req, tree)
            release(req, tree)  # the leaked row is not retried
            totals.append(owner.occ3_quarantined_tokens)
            self.assertEqual(tree.calls, [])
            self.assertEqual(tree.freed_pages, set())
            self.assertEqual(tree.req_to_token_pool.frees, 1)
            self.assertTrue(req.occ3_kv_quarantined)
            self.assertFalse(req.occ3_kv_released)
            self.assertFalse(req.kv.holds_kv)
            self.assertEqual(req.kv.kv_allocated_len, 0)
        # Page-rounded allocation length, counted once per quarantined row.
        self.assertEqual(totals, [192, 128])
        self.assertEqual([(record[0], record[3]) for record in log.records],
                         [("warning", "duplicate"), ("warning", "range")])
        self.assertEqual([record[4:7] for record in log.records], [(3, 192, 192), (2, 128, 128)])
        self.assertEqual(len(self.fence_calls), 3)

    def idle_fixture(self, capacity=1024, page=64):
        """Real manifest-applied on_idle, flush_cache and quarantine methods.

        The invariant checker is a stand-in for the stock full-pool identity
        (available + evictable + protected + session_held + uncached == total).
        Strict idle mode is on, so any reported leak raises.
        """
        from contextlib import nullcontext
        log, reports = Log(), []
        stats = NS(full_available_size=capacity, full_evictable_size=0)
        def check_all_pools(ps, uncached=0):
            total = ps.full_available_size + ps.full_evictable_size + uncached
            return total != capacity, [f"[full] total={capacity}, available={ps.full_available_size}, uncached={uncached}"]
        def report(pool, message):
            reports.append((pool, message))
            raise ValueError(message)  # strict idle: SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=True

        class IdleChecker:
            """Stock stand-in; the idle path never asks for busy uncached sizes.

            A real class, not a lambda: the patched init_invariant_checker
            subclasses SchedulerInvariantChecker so the busy check can add the
            quarantine counter to the stock uncached sum.
            """

            def __init__(self, **kwargs):
                self._check_all_pools = check_all_pools
                self._report_leak = report
                self._check_req_pool = lambda: None
                self._check_tree_cache = lambda: None

            def _get_total_uncached_sizes(self):
                return 0, 0

        checker = IdleChecker()
        allocator = NS(page_size=page, verify_byte_accounting=lambda: [], clears=0)
        def clear():
            allocator.clears += 1
            stats.full_available_size = capacity
        allocator.clear = clear
        metrics = NS(record_scheduler_idle=lambda: None, _maybe_log_idle_metrics=lambda: None,
                     record_scheduler_active=lambda: None, reset_metrics=lambda: None,
                     is_stats_logging_rank=False)
        owner = NS(token_to_kv_pool_allocator=allocator, max_total_num_tokens=capacity,
                   invariant_checker=checker, pool_stats_observer=NS(get_pool_stats=lambda: stats),
                   scheduler_stage_metrics=NS(record=lambda stage: nullcontext()),
                   maybe_send_health_check_signal=lambda: None, is_fully_idle=lambda **kw: True,
                   enable_unified_memory=False, enable_hisparse=False, disaggregation_mode="null",
                   metrics_reporter=metrics, kv_events_publisher=NS(publish_kv_events=lambda: None),
                   new_token_ratio_tracker=NS(reset=lambda: None), publish_load_snapshot=lambda **kw: None,
                   load_publisher=NS(publish_load_stat=lambda *a, **kw: None),
                   load_inquirer=NS(get_loads=lambda: None), maybe_sleep_on_idle=lambda: None,
                   _engine_paused=False, draft_worker=None, grammar_manager=NS(clear=lambda: None),
                   tree_cache=NS(reset=lambda: None),
                   req_to_token_pool=NS(clear=lambda: None, reset_aux_cache_allocator=lambda: None),
                   is_hybrid_swa=False, is_hybrid_ssm=False, page_size=page, full_tokens_per_layer=None,
                   swa_tokens_per_layer=None, kv_shard_widening=1)
        fn = self.sources.function
        init = fn("managers/scheduler.py", "init_invariant_checker", "Scheduler",
                  SchedulerInvariantChecker=IdleChecker)
        init(owner)
        # The patched init wraps the stock checker in the quarantine-aware
        # subclass; the idle stand-in methods still come from IdleChecker.
        assert type(owner.invariant_checker) is not IdleChecker
        assert isinstance(owner.invariant_checker, IdleChecker)
        quarantine = fn("managers/scheduler.py", "_occ3_quarantine_private_kv", "Scheduler", logger=log)
        owner._occ3_quarantine_private_kv = types.MethodType(quarantine, owner)
        idle = fn("managers/scheduler.py", "on_idle", "Scheduler",
                  DisaggregationMode=NS(DECODE="decode"), SCHEDULER_STAGE_SANITY_CHECK_CACHE="cache")
        flush = fn("managers/scheduler.py", "flush_cache", "Scheduler", logger=log,
                   logging=log, current_platform=NS(empty_cache=lambda: None))
        def withhold(allocated, fault="duplicate"):
            req = NS(rid=f"r{allocated}", occ3_kv_quarantined=False)
            owner._occ3_quarantine_private_kv(req, allocated, fault)
            stats.full_available_size -= ((allocated + page - 1) // page) * page
            return req
        return NS(owner=owner, stats=stats, log=log, reports=reports, idle=idle,
                  flush=flush, withhold=withhold, allocator=allocator)

    def test_quarantine_balances_strict_idle_identity(self):
        f = self.idle_fixture()
        self.assertEqual(f.owner.occ3_quarantined_tokens, 0)
        req = f.withhold(138)
        self.assertTrue(req.occ3_kv_quarantined)
        self.assertEqual(f.owner.occ3_quarantined_tokens, 192)
        f.idle(f.owner)  # strict: would raise on any reported leak
        self.assertEqual(f.reports, [])
        self.assertEqual(f.log.records[0][:4], ("warning", f.log.records[0][1], "r138", "duplicate"))
        self.assertEqual(f.log.records[0][4:7], (3, 192, 192))

    def test_quarantine_does_not_hide_an_unrelated_leak(self):
        f = self.idle_fixture()
        f.withhold(64)
        f.stats.full_available_size -= 64  # a real leak outside quarantine
        with self.assertRaises(ValueError):
            f.idle(f.owner)
        self.assertEqual([pool for pool, _ in f.reports], ["pool"])
        self.assertIn("uncached=64", f.reports[0][1])

    def test_quarantine_error_once_above_five_percent_and_flush_resets(self):
        f = self.idle_fixture(capacity=1024)
        f.withhold(64)  # 6.25% of capacity
        f.withhold(64)
        errors = [record for record in f.log.records if record[0] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(f.owner.occ3_quarantined_tokens, 128)
        self.assertTrue(f.flush(f.owner))
        self.assertEqual(f.allocator.clears, 1)
        self.assertEqual(f.owner.occ3_quarantined_tokens, 0)
        self.assertFalse(f.owner._occ3_quarantine_error_logged)
        f.idle(f.owner)
        self.assertEqual(f.reports, [])

    def test_quarantine_counter_kept_when_flush_declines(self):
        f = self.idle_fixture()
        f.withhold(64)
        f.owner.is_fully_idle = lambda **kw: False
        f.owner.waiting_queue, f.owner.running_batch = [1], NS(reqs=[])
        self.assertFalse(f.flush(f.owner))
        self.assertEqual(f.allocator.clears, 0)
        self.assertEqual(f.owner.occ3_quarantined_tokens, 64)

    def busy_fixture(self, capacity=1024, page=64):
        """Real manifest-applied init_invariant_checker with a stock-like checker.

        The stand-in mirrors the stock busy identity: uncached tokens come
        from active batches only (_get_total_uncached_sizes), and the full
        pool must balance available + evictable + uncached == total. The
        patched init wraps the stand-in in the quarantine-aware subclass, so
        self_check_during_busy sees quarantined pages as uncached, exactly
        like the patched idle path.
        """
        from contextlib import nullcontext
        log = Log()
        stats = NS(full_available_size=capacity, full_evictable_size=0)
        observer = NS(get_pool_stats=lambda: stats)

        class StockChecker:
            """Stand-in for the stock SchedulerInvariantChecker busy path."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.active_uncached = 0

            def _get_total_uncached_sizes(self):
                return self.active_uncached, 0

            def _check_full_pool(self, ps, uncached=0):
                total = ps.full_available_size + ps.full_evictable_size + uncached
                leak = total != self.kwargs["max_total_num_tokens"]
                return leak, (f"[full] total={self.kwargs['max_total_num_tokens']}, "
                              f"available={ps.full_available_size}, uncached={uncached}")

            def self_check_during_busy(self):
                full_uncached, _ = self._get_total_uncached_sizes()
                return self._check_full_pool(self.kwargs["pool_stats_observer"].get_pool_stats(),
                                             uncached=full_uncached)

        allocator = NS(page_size=page, verify_byte_accounting=lambda: [], clears=0)
        owner = NS(token_to_kv_pool_allocator=allocator, max_total_num_tokens=capacity,
                   tp_group=NS(ranks=[0]), tp_cpu_group=None,
                   pool_stats_observer=observer,
                   scheduler_stage_metrics=NS(record=lambda stage: nullcontext()),
                   is_hybrid_swa=False, is_hybrid_ssm=False,
                   disaggregation_mode="null", page_size=page,
                   full_tokens_per_layer=None, swa_tokens_per_layer=None,
                   kv_shard_widening=1, tree_cache=NS(),
                   req_to_token_pool=NS(), scheduler_stage_metrics_recorder=None)
        fn = self.sources.function
        init = fn("managers/scheduler.py", "init_invariant_checker", "Scheduler",
                  SchedulerInvariantChecker=StockChecker)
        init(owner)
        checker = owner.invariant_checker
        # The patched init produced the quarantine-aware subclass of the stock
        # stand-in, not the stand-in class itself.
        self.assertIsNot(type(checker), StockChecker)
        self.assertIsInstance(checker, StockChecker)
        quarantine = fn("managers/scheduler.py", "_occ3_quarantine_private_kv", "Scheduler", logger=log)
        owner._occ3_quarantine_private_kv = types.MethodType(quarantine, owner)

        def withhold(allocated, fault="duplicate"):
            req = NS(rid=f"r{allocated}", occ3_kv_quarantined=False)
            owner._occ3_quarantine_private_kv(req, allocated, fault)
            stats.full_available_size -= ((allocated + page - 1) // page) * page
            return req

        return NS(owner=owner, checker=checker, stats=stats, log=log,
                  withhold=withhold, allocator=allocator)

    def test_busy_check_counts_quarantined_pages(self):
        f = self.busy_fixture()
        self.assertEqual(f.owner.occ3_quarantined_tokens, 0)
        # No quarantine: ordinary active-batch uncached tokens balance (the
        # pages are allocated, so they are missing from available).
        f.checker.active_uncached = 64
        f.stats.full_available_size -= 64
        leak, message = f.checker.self_check_during_busy()
        self.assertFalse(leak)
        self.assertIn("uncached=64", message)
        # A quarantined row releases its KV claim while its pages stay
        # allocated: without quarantine accounting the strict busy check
        # would report exactly those tokens as a leak.
        f.withhold(138)  # 3 pages of 64 -> 192 tokens
        self.assertEqual(f.owner.occ3_quarantined_tokens, 192)
        leak, message = f.checker.self_check_during_busy()
        self.assertFalse(leak)
        # Active-batch uncached and quarantined tokens add up together.
        self.assertIn("uncached=256", message)
        # The counter is read live: a second quarantined row is counted on the
        # very next busy check without re-running init_invariant_checker.
        f.withhold(70)  # 2 pages of 64 -> 128 tokens
        leak, message = f.checker.self_check_during_busy()
        self.assertFalse(leak)
        self.assertIn("uncached=384", message)

    def test_busy_check_does_not_hide_an_unrelated_leak(self):
        f = self.busy_fixture()
        f.withhold(64)
        f.stats.full_available_size -= 64  # a real leak outside quarantine
        leak, message = f.checker.self_check_during_busy()
        self.assertTrue(leak)
        self.assertIn("uncached=64", message)

    def test_busy_check_quarantine_accounting_survives_flush_reset(self):
        f = self.busy_fixture()
        f.withhold(64)
        leak, _ = f.checker.self_check_during_busy()
        self.assertFalse(leak)
        # flush_cache reclaims quarantined pages and zeroes the counter; the
        # busy identity balances again without the quarantine term.
        f.owner.occ3_quarantined_tokens = 0
        f.stats.full_available_size += 64
        f.checker.active_uncached = 0
        leak, message = f.checker.self_check_during_busy()
        self.assertFalse(leak)
        self.assertIn("uncached=0", message)

    def test_private_release_tp_vote_quarantines_all_or_frees_all(self):
        # Sequential rank stand-ins: each reducer supplies the known peer bit.
        # This proves decision wiring, not distributed progress or CUDA behavior.
        for faulty_rank in (None, 0, 1):
            calls, logs, requests, trees = [], [], [], []
            group = object()
            for rank in range(2):
                log = Log()
                row = page_row([7, 3, 7] if rank == faulty_rank else [7, 3, 12], 138)
                req, tree = request(138, 138), paged_cache(row)
                req.occ3_scheduler = self.quarantine_owner(tree, log, ranks=(0, 1), group=group)
                max_op = object()
                def reduce_flag(flag, op, group, rank=rank):
                    self.assertIs(op, max_op)
                    self.assertIs(group, req.occ3_scheduler.tp_cpu_group)
                    self.assertEqual(flag.dtype, np.int32)
                    self.assertEqual(flag.device, "cpu")
                    self.assertEqual(flag.tolist(), [int(rank == faulty_rank)])
                    calls.append(rank)
                    flag[0] = int(faulty_rank is not None)
                torch_stub = NS(tensor=tensor, int32=np.int32,
                                distributed=NS(all_reduce=reduce_flag, ReduceOp=NS(MAX=max_op)))
                predicate = self.sources.function("mem_cache/common.py", "occ3_private_row_fault", torch=NP_TORCH)
                decision = self.sources.function("mem_cache/common.py", "occ3_private_release_fault",
                                                 occ3_private_row_fault=predicate, torch=torch_stub)
                release = self.sources.function("mem_cache/common.py", "occ3_release_private_kv",
                                                occ3_private_release_fault=decision, logger=log)
                release(req, tree)
                requests.append(req); trees.append(tree); logs.append(log)
            self.assertEqual(calls, [0, 1])
            for rank, (req, tree, log) in enumerate(zip(requests, trees, logs)):
                self.assertEqual(req.occ3_kv_quarantined, faulty_rank is not None)
                self.assertEqual(tree.req_to_token_pool.frees, 1)
                self.assertEqual(bool(tree.calls), faulty_rank is None)
                # Every rank counts the same quarantine, so capacity stays equal.
                self.assertEqual(req.occ3_scheduler.occ3_quarantined_tokens,
                                 0 if faulty_rank is None else 192)
                if faulty_rank is not None:
                    self.assertEqual(log.records[0][3], "duplicate" if rank == faulty_rank else "peer")
                else:
                    self.assertEqual(log.records, [])

    def test_private_release_vote_empty_and_predicate_error_participate(self):
        for allocated, expected_local in ((0, 0), (8, 1)):
            calls = []
            def predicate(*args):
                raise ValueError("predicate failed")
            def reduce_flag(flag, **kwargs):
                calls.append(int(flag[0]))
                flag[0] = 1  # A faulty peer also makes the empty local row decline.
            torch_stub = NS(tensor=tensor, int32=np.int32,
                            distributed=NS(all_reduce=reduce_flag, ReduceOp=NS(MAX="max")))
            req = request(allocated, allocated)
            req.occ3_scheduler = NS(tp_group=NS(ranks=[0, 1]), tp_cpu_group=object())
            decision = self.sources.function("mem_cache/common.py", "occ3_private_release_fault",
                                             occ3_private_row_fault=predicate, torch=torch_stub)
            self.assertEqual(decision(req, Cache(), allocated), "check_failed" if allocated else "peer")
            self.assertEqual(calls, [expected_local])

    def test_private_release_vote_single_rank_has_no_collective(self):
        # No tensor/distributed attribute: even constructing a flag would fail.
        for local_fault in (None, "range"):
            decision = self.sources.function("mem_cache/common.py", "occ3_private_release_fault",
                                             occ3_private_row_fault=lambda *args: local_fault,
                                             torch=NS())
            self.assertEqual(decision(request(), Cache(), 8), local_fault)

    def test_private_release_rejects_writer_shaped_duplicate_and_out_of_range_rows(self):
        """Execute manifest-applied release and row check, not a predicate stub."""
        fault = self.sources.function("mem_cache/common.py", "occ3_private_row_fault", torch=NP_TORCH)
        good = page_row([7, 3], 128).astype(np.int32)
        # Match the writer's int64 load / int32 store, with enough backing for
        # every load. The second half is padding, not an out-of-allocation read.
        backing = np.concatenate([good, np.zeros_like(good)])
        writer_shaped = backing.view(np.int64).astype(np.int32)
        cases = [("layout", writer_shaped),
                 ("duplicate", page_row([7, 3, 7], 138)),
                 ("range", page_row([7, 17], 100))]
        for reason, row in cases:
            with self.subTest(reason=reason):
                log = Log()
                decision = self.sources.function("mem_cache/common.py", "occ3_private_release_fault",
                                                 occ3_private_row_fault=fault, torch=NP_TORCH)
                release = self.sources.function("mem_cache/common.py", "occ3_release_private_kv",
                                                occ3_private_release_fault=decision, logger=log)
                req, tree = request(len(row), len(row)), paged_cache(row)
                req.occ3_scheduler = owner = self.quarantine_owner(tree, log)
                release(req, tree)
                self.assertEqual(owner.occ3_quarantined_tokens, ((len(row) + 63) // 64) * 64)
                self.assertEqual(tree.calls, [], "untrusted pages must not reach free_kv_row")
                self.assertEqual(tree.freed_pages, set())
                self.assertEqual(tree.req_to_token_pool.frees, 1)
                self.assertTrue(req.occ3_kv_quarantined)
                self.assertFalse(req.occ3_kv_released)
                self.assertFalse(req.kv.holds_kv)
                self.assertEqual(req.kv.kv_allocated_len, 0)
                self.assertEqual(req.kv.kv_committed_len, 0)
                self.assertEqual(len(req.prefix_indices), 0)
                self.assertEqual([(record[0], record[3]) for record in log.records],
                                 [("warning", reason)])
                release(req, tree)
                self.assertEqual(tree.calls, [])
                self.assertEqual(tree.req_to_token_pool.frees, 1)
                self.assertEqual(len(log.records), 1)
        self.assertEqual(len(self.fence_calls), len(cases))

    @unittest.skipUnless(os.environ.get("OCC3_ROOT_IMAGE_GPU_TESTS") == "1",
                         "root-only actual Triton chunk-two writer test; allocates CUDA tensors")
    def test_root_only_private_chunk_two_actual_triton_writer(self):
        """Real writer plus manifest-applied stash; no model forward or TP proof."""
        try:
            import torch
            import triton
        except ImportError as exc:
            self.skipTest(f"real writer dependency unavailable: {exc.name}")
        if not torch.cuda.is_available():
            self.skipTest("real writer requires an available CUDA GPU")
        from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton

        _, _, stash, _ = self.functions(torch_module=torch)
        # Chunk one ends within a page. Chunk two retains its private prefix and
        # appends fresh slots, as alloc_for_extend does (one request, row 3).
        committed, extension, capacity = 70, 19, 128
        req, tree = request(committed, committed, end=committed), Cache()
        table = torch.full((4, capacity), -1, dtype=torch.int32, device="cuda")
        first = torch.arange(64, 64 + committed, dtype=torch.int32, device="cuda")
        table[3, :committed] = first
        tree.req_to_token_pool.req_to_token = table
        req.prefix_indices = torch.empty(0, dtype=torch.int64, device="cuda")
        stash(req, tree, chunked=True)
        suffix = torch.arange(64 + committed, 64 + committed + extension,
                              dtype=torch.int64, device="cuda")
        indices = torch.tensor([3], dtype=torch.int64, device="cuda")
        pre_lens = torch.tensor([committed], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([committed + extension], dtype=torch.int64, device="cuda")
        extend_lens = torch.tensor([extension], dtype=torch.int64, device="cuda")

        def write_chunk_two(prefix):
            # Always reserve 2 * committed elements. Even an old int32 stash
            # can be read as committed int64 elements entirely in-bounds.
            backing = torch.zeros(2 * committed, dtype=prefix.dtype, device="cuda")
            backing[:committed].copy_(prefix)
            pointers = torch.tensor([backing.data_ptr()], dtype=torch.uint64, device="cuda")
            result = table.clone()
            write_req_to_token_pool_triton[(1,)](
                result, indices, pointers, pre_lens, seq_lens, extend_lens,
                suffix, result.stride(0),
            )
            torch.cuda.synchronize()  # retain backing until all pointer loads finish
            return result

        expected = table.clone()
        expected[3, committed:committed + extension] = suffix.to(torch.int32)
        # This assertion also rejects an old .clone() stash without an unsafe
        # read: its padded int32 prefix produces every-other-slot values.
        self.assertTrue(torch.equal(write_chunk_two(req.prefix_indices), expected))
        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        narrow = write_chunk_two(first)
        self.assertFalse(torch.equal(narrow[3, :committed], first))
        backing = torch.cat([first, torch.zeros_like(first)])
        self.assertTrue(torch.equal(narrow[3, :committed], backing.view(torch.int64).to(torch.int32)))
        self.assertTrue(torch.equal(narrow[3, committed:], expected[3, committed:]))
        self.assertTrue(torch.equal(narrow[:3], expected[:3]))

    def test_retraction_fence_failure_never_releases(self):
        worker=types.ModuleType("occ3_sglang.latent_worker");calls=[]
        def fail(*a):raise RuntimeError("poisoned runtime fence failed")
        worker.abort_request=fail
        fn=self.sources.function("managers/schedule_batch.py","release_req",
            release_kv_cache=lambda *a,**kw:calls.append("free"))
        req=request();req.occ3_force_eager=True
        with patch.dict("sys.modules",{worker.__name__:worker}):
            with self.assertRaisesRegex(RuntimeError,"cleanup_uncertain"):
                fn(req=req,remaing_req_count=0,req_to_token_pool=None,token_to_kv_pool_allocator=None,
                   tree_cache=None,hisparse_coordinator=None)
        self.assertTrue(req.occ3_kv_quarantined)
        self.assertEqual(calls,[])
        self.assertTrue(req.kv.holds_kv)

    def _root_actual_allocator_roundtrip(self,device):
        """Root-only actual Torch/native paged allocator; no model or NCCL proof."""
        import hashlib
        from pathlib import Path
        try:
            import torch
        except ImportError:
            self.skipTest("real allocator test requires Torch")
        from sglang.srt.managers import schedule_batch
        from sglang.srt.mem_cache import common
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        for path,module in [("mem_cache/common.py",common),("managers/schedule_batch.py",schedule_batch)]:
            expected=next(f["after_sha256"] for f in self.sources.manifest["files"] if f["path"]==path)
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),expected)
        for page in (1,4,16):
            for committed,allocated in [(1,1),(3,4),(5,8),(15,17),(31,33)]:
                allocator=PagedTokenToKVPoolAllocator(size=128,page_size=page,dtype=torch.bfloat16,
                    device=device,kvcache=None,need_sort=False)
                allocator.debug_mode=True
                reserved=((allocated+page-1)//page)*page
                # ReqToTokenPool stores int32, unlike the allocator's int64 output.
                row=allocator.alloc(reserved).to(dtype=torch.int32).reshape(1,-1)
                self.assertEqual(row.dtype,torch.int32)
                self.assertEqual(allocator.available_size(),128-reserved)
                kv=schedule_batch.ReqKvInfo(req_pool_idx=0,kv_committed_len=committed,kv_allocated_len=allocated)
                releases=[]
                def free_row(req):
                    self.assertIsNotNone(req.kv.req_pool_idx)
                    releases.append(req.kv.req_pool_idx)
                    req.kv.req_pool_idx=None
                tree=NS(root_node=object(),req_to_token_pool=NS(req_to_token=row,free=free_row),
                        token_to_kv_pool_allocator=allocator)
                tree.root_node_handle=lambda extra_key=None:BasePrefixCache.root_node_handle(tree,extra_key)
                tree.free_kv_row=lambda kv,ranges:BasePrefixCache.free_kv_row(tree,kv,ranges)
                req=NS(occ3_private_prefix=True,skip_radix_cache_insert=True,kv=kv,
                       prefix_indices=torch.empty(0,dtype=torch.int64,device=device),
                       extend_range=NS(end=committed),extra_key=None,occ3_kv_released=False,
                       occ3_scheduler=object(),occ3_kv_quarantined=False,occ3_kv_release_started=False)
                common.maybe_cache_unfinished_req(req,tree)
                self.assertEqual(req.prefix_indices.dtype,torch.int64)  # alloc_for_extend writer ABI
                self.assertTrue(torch.equal(req.prefix_indices,row[0,:committed].to(torch.int64)))
                common.release_kv_cache(req,tree)
                common.release_kv_cache(req,tree)
                self.assertEqual(releases,[0])
                self.assertEqual(allocator.available_size(),128)
                self.assertEqual(len(torch.unique(allocator.free_pages)),128//page)
                self.assertEqual(kv.cache_protected_len,0)
                self.assertTrue(kv.is_kv_released)
        if device=="cuda":torch.cuda.synchronize()

    @unittest.skipUnless(os.environ.get("OCC3_ROOT_IMAGE_CPU_TESTS")=="1","root-only exact installed Torch/native allocator CPU test")
    def test_root_only_native_paged_allocator_cpu(self):
        self._root_actual_allocator_roundtrip("cpu")

    @unittest.skipUnless(os.environ.get("OCC3_ROOT_IMAGE_GPU_TESTS")=="1","root-only exact installed CUDA/native allocator test; allocates device tensors")
    def test_root_only_native_paged_allocator_cuda(self):
        try:
            import torch
        except ImportError:
            self.skipTest("real allocator test requires Torch")
        self.assertTrue(torch.cuda.is_available(),"explicit root GPU gate requires actual CUDA")
        self._root_actual_allocator_roundtrip("cuda")


if __name__ == "__main__":
    unittest.main()
