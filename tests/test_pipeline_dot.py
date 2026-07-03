"""Regression tests for the cross-iteration pipelined dot (review findings)."""

import pytest
import torch

import newt
import newt.language as nl

DEVICE = "cuda"


def test_main_plus_tail_loops_same_acc():
    """Two pipelined dot sites on one accumulator: the second loop running
    ZERO iterations must not drop the first site's deferred mma."""
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, K, FULL, TAIL,
          BM: nl.constexpr, BN: nl.constexpr, BK: nl.constexpr):
        om = nl.arange(0, BM)
        on = nl.arange(0, BN)
        ok = nl.arange(0, BK)
        a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
        b_ptrs = b_ptr + ok[:, None] * BN + on[None, :]
        acc = nl.zeros((BM, BN), dtype=nl.float32)
        for i in range(FULL):
            acc = nl.dot(nl.load(a_ptrs), nl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK * BN
        for t in range(TAIL):
            kmask = (t * BK + ok) < (K - FULL * BK)
            at = nl.load(a_ptrs, mask=kmask[None, :], other=0.0)
            bt = nl.load(b_ptrs, mask=kmask[:, None], other=0.0)
            acc = nl.dot(at, bt, acc)
        nl.store(o_ptr + om[:, None] * BN + on[None, :], acc)

    BM, BN, BK = 64, 64, 32
    for K in (96, 80):  # TAIL = 0 and TAIL = 1
        FULL, rem = K // BK, K % BK
        TAIL = 1 if rem else 0
        a = torch.randn(BM, K, device=DEVICE, dtype=torch.float16) * 0.3
        b = torch.randn(K, BN, device=DEVICE, dtype=torch.float16) * 0.3
        o = torch.empty(BM, BN, device=DEVICE, dtype=torch.float32)
        k[(1,)](a, b, o, K, FULL, TAIL, BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        assert torch.allclose(o, ref, rtol=2e-2, atol=2e-2), f"K={K}"


def test_overwrite_unread_acc():
    """Overwriting an accumulator with a pending deferred mma (never read)
    must not let the stale tile fire into the new value."""
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, K, BM: nl.constexpr, BN: nl.constexpr,
          BK: nl.constexpr):
        om = nl.arange(0, BM)
        on = nl.arange(0, BN)
        ok = nl.arange(0, BK)
        a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
        b_ptrs = b_ptr + ok[:, None] * BN + on[None, :]
        c2 = nl.dot(nl.load(a_ptrs), nl.load(b_ptrs))  # first K-tile only
        acc = nl.zeros((BM, BN), dtype=nl.float32)
        for i in range(K // BK):
            acc = nl.dot(nl.load(a_ptrs), nl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK * BN
        acc = c2  # discard the pipelined result without reading it
        nl.store(o_ptr + om[:, None] * BN + on[None, :], acc + 0.0)

    BM, BN, BK, K = 64, 64, 32, 96
    a = torch.randn(BM, K, device=DEVICE, dtype=torch.float16) * 0.3
    b = torch.randn(K, BN, device=DEVICE, dtype=torch.float16) * 0.3
    o = torch.empty(BM, BN, device=DEVICE, dtype=torch.float32)
    k[(1,)](a, b, o, K, BM=BM, BN=BN, BK=BK)
    torch.cuda.synchronize()
    ref = a[:, :BK].float() @ b[:BK, :].float()
    assert torch.allclose(o, ref, rtol=2e-2, atol=2e-2)


def test_oversized_ring_falls_back():
    """A config whose double-buffer ring exceeds the smem budget must fall
    back to the chunked path instead of failing to compile."""
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, K,
          BM: nl.constexpr, BN: nl.constexpr, BK: nl.constexpr):
        om = nl.arange(0, BM)
        on = nl.arange(0, BN)
        ok = nl.arange(0, BK)
        a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
        b_ptrs = b_ptr + ok[:, None] * BN + on[None, :]
        acc = nl.zeros((BM, BN), dtype=nl.float32)
        for i in range(K // BK):
            acc = nl.dot(nl.load(a_ptrs), nl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK * BN
        nl.store(o_ptr + om[:, None] * BN + on[None, :], acc)

    # fp32/tf32 at 128x128x64: ring alone would be ~137KB
    BM, BN, BK, K = 128, 128, 64, 128
    a = torch.randn(BM, K, device=DEVICE) * 0.3
    b = torch.randn(K, BN, device=DEVICE) * 0.3
    o = torch.empty(BM, BN, device=DEVICE)
    k[(1,)](a, b, o, K, BM=BM, BN=BN, BK=BK, num_warps=8)
    torch.cuda.synchronize()
    assert torch.allclose(o, a @ b, rtol=2e-2, atol=2e-1)


def test_store_inside_k_loop():
    """An in-loop store (memory mutation) drains in-flight tiles first."""
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, dbg_ptr, K,
          BM: nl.constexpr, BN: nl.constexpr, BK: nl.constexpr):
        om = nl.arange(0, BM)
        on = nl.arange(0, BN)
        ok = nl.arange(0, BK)
        a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
        b_ptrs = b_ptr + ok[:, None] * BN + on[None, :]
        acc = nl.zeros((BM, BN), dtype=nl.float32)
        for i in range(K // BK):
            acc = nl.dot(nl.load(a_ptrs), nl.load(b_ptrs), acc)
            nl.store(dbg_ptr + om, om.to(nl.float32))  # unrelated in-loop store
            a_ptrs += BK
            b_ptrs += BK * BN
        nl.store(o_ptr + om[:, None] * BN + on[None, :], acc)

    BM, BN, BK, K = 64, 64, 32, 128
    a = torch.randn(BM, K, device=DEVICE, dtype=torch.float16) * 0.3
    b = torch.randn(K, BN, device=DEVICE, dtype=torch.float16) * 0.3
    o = torch.empty(BM, BN, device=DEVICE, dtype=torch.float32)
    dbg = torch.empty(BM, device=DEVICE)
    k[(1,)](a, b, o, dbg, K, BM=BM, BN=BN, BK=BK)
    torch.cuda.synchronize()
    assert torch.allclose(o, a.float() @ b.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
