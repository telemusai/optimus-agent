"""Exact C cache methods with bounded CPU index/allocator stubs; no GPU proof."""
from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from test_latent_ipc_patch import PinnedCase


class Indices(list):
    """Index-only stub, NOT a Torch tensor or mathematical reference."""
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Indices(result) if isinstance(key, slice) else result
    def clone(self): return Indices(self)
    def numel(self): return len(self)


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
              occ3_scheduler=object(), occ3_kv_quarantined=False, occ3_kv_release_started=False)


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

    def functions(self):
        path = "mem_cache/common.py"
        match = self.sources.function(path, "occ3_private_match", MatchResult=lambda **kwargs: NS(**kwargs))
        release = self.sources.function(path, "occ3_release_private_kv")
        cache = self.sources.function(path, "maybe_cache_unfinished_req", occ3_private_match=match)
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
        match, _, cache, _ = self.functions()
        req, tree = request(), Cache()
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        tree.req_to_token_pool.req_to_token = torch.arange(64, dtype=torch.int64).repeat(4, 1)
        cache(req, tree)
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
        import torch
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
                row=allocator.alloc(reserved).reshape(1,-1)
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
                self.assertTrue(torch.equal(req.prefix_indices,row[0,:committed]))
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
        import torch
        self.assertTrue(torch.cuda.is_available(),"explicit root GPU gate requires actual CUDA")
        self._root_actual_allocator_roundtrip("cuda")


if __name__ == "__main__":
    unittest.main()
