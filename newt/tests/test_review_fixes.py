"""Regression tests for the adversarial-review findings (all GPU-confirmed)."""

import pytest
import torch

import newt
import newt.language as nl

DEVICE = "cuda"


def test_hoisted_var_after_loop():
    @newt.jit
    def k(x_ptr, y_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        for i in range(2):
            t = nl.load(x_ptr + offs) + i
        nl.store(y_ptr + offs, t)

    x = torch.randn(64, device=DEVICE)
    y = torch.empty_like(x)
    k[(1,)](x, y, BLOCK=64)
    assert torch.equal(y, x + 1)


def test_hoisted_var_after_if():
    @newt.jit
    def k(x_ptr, y_ptr, flag, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        if flag == 1:
            t = nl.load(x_ptr + offs) * 2
        else:
            t = nl.load(x_ptr + offs) * 3
        nl.store(y_ptr + offs, t)

    x = torch.randn(64, device=DEVICE)
    y = torch.empty_like(x)
    k[(1,)](x, y, 1, BLOCK=64)
    assert torch.equal(y, x * 2)
    k[(1,)](x, y, 0, BLOCK=64)
    assert torch.equal(y, x * 3)


def test_no_storage_aliasing():
    @newt.jit
    def k(x_ptr, out_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        x = nl.load(x_ptr + offs)
        for i in range(2):
            prev = x
            x = x + 1.0
            nl.store(out_ptr + i * BLOCK + offs, prev)

    B = 64
    x = torch.zeros(B, device=DEVICE)
    out = torch.empty(2 * B, device=DEVICE)
    k[(1,)](x, out, BLOCK=B)
    assert torch.all(out[:B] == 0.0), "prev aliased its source"
    assert torch.all(out[B:] == 1.0)


def test_dot_after_store_uses_original_data():
    """Re-using dot-consumed loads after a store must not silently re-read."""
    @newt.jit
    def k(x_ptr, y_ptr, c1_ptr, c2_ptr, N: nl.constexpr):
        r = nl.arange(0, N)
        idx = r[:, None] * N + r[None, :]
        x = nl.load(x_ptr + idx)
        y = nl.load(y_ptr + idx)
        c1 = nl.dot(x, y)
        nl.store(c1_ptr + idx, c1)
        nl.store(x_ptr + idx, nl.zeros((N, N), dtype=nl.float16))
        c2 = nl.dot(x, y)
        nl.store(c2_ptr + idx, c2)

    N = 16
    x = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    y = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    c1 = torch.empty(N, N, device=DEVICE, dtype=torch.float32)
    c2 = torch.empty_like(c1)
    with pytest.raises(newt.CompileError, match="consumed"):
        k[(1,)](x, y, c1, c2, N=N)


def test_bool_reductions():
    @newt.jit
    def k(x_ptr, y_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        m = nl.load(x_ptr + offs) > 100.0
        nl.store(y_ptr, nl.max(m).to(nl.int32))
        nl.store(y_ptr + 1, nl.sum(m).to(nl.int32))

    x = torch.zeros(64, device=DEVICE)
    x[3] = x[10] = x[40] = 200.0
    y = torch.empty(2, device=DEVICE, dtype=torch.int32)
    k[(1,)](x, y, BLOCK=64)
    assert y[0].item() == 1 and y[1].item() == 3
    x.zero_()
    k[(1,)](x, y, BLOCK=64)
    assert y[0].item() == 0 and y[1].item() == 0


def test_scalar_atomic_once_per_program():
    @newt.jit
    def k(out_ptr):
        nl.atomic_add(out_ptr, 1.0)

    o = torch.zeros(1, device=DEVICE)
    k[(10,)](o, num_warps=4)
    torch.cuda.synchronize()
    assert o.item() == 10.0


def test_atomic_max_float():
    @newt.jit
    def k(x_ptr, out_ptr, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        nl.atomic_max(out_ptr, nl.max(nl.load(x_ptr + offs)))

    x = torch.randn(4096, device=DEVICE)
    o = torch.full((1,), float("-inf"), device=DEVICE)
    k[(4,)](x, o, BLOCK=1024)
    torch.cuda.synchronize()
    assert o.item() == x.max().item()


def test_autotune_reset_to_zero_first_call():
    @newt.autotune(
        configs=[newt.Config({"BLOCK": 64}), newt.Config({"BLOCK": 64}, num_warps=2)],
        key=["n"], reset_to_zero=["out_ptr"])
    @newt.jit
    def rmw(x_ptr, out_ptr, n, BLOCK: nl.constexpr):
        pid = nl.program_id(0)
        offs = pid * BLOCK + nl.arange(0, BLOCK)
        m = offs < n
        o = nl.load(out_ptr + offs, mask=m)
        nl.store(out_ptr + offs, o + nl.load(x_ptr + offs, mask=m), mask=m)

    x = torch.ones(256, device=DEVICE)
    out = torch.zeros(256, device=DEVICE)
    rmw[lambda meta: (newt.cdiv(256, meta["BLOCK"]),)](x, out, 256)
    torch.cuda.synchronize()
    assert out[0].item() == 1.0  # first (tuning) call must not double-accumulate


def test_grid_meta_has_runtime_args():
    @newt.jit
    def k(x_ptr, y_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        m = offs < n
        nl.store(y_ptr + offs, nl.load(x_ptr + offs, mask=m) + 1, mask=m)

    n = 1000
    x = torch.randn(n, device=DEVICE)
    y = torch.empty_like(x)
    k[lambda META: (newt.cdiv(META["n"], META["BLOCK"]),)](x, y, n, BLOCK=256)
    assert torch.equal(y, x + 1)


def test_zero_grid_is_noop():
    @newt.jit
    def k(x_ptr, BLOCK: nl.constexpr):
        nl.store(x_ptr + nl.arange(0, BLOCK), 1.0)

    x = torch.zeros(64, device=DEVICE)
    k[(0,)](x, BLOCK=64)  # must not raise
    torch.cuda.synchronize()
    assert torch.all(x == 0)


def test_scalar_ptr_block_mask_rejected():
    @newt.jit
    def k(x_ptr, y_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        m = offs < 4
        v = nl.load(x_ptr, mask=m, other=7.0)
        nl.store(y_ptr + offs, nl.zeros((BLOCK,), dtype=nl.float32) + v, mask=m)

    x = torch.tensor([42.0], device=DEVICE)
    y = torch.empty(64, device=DEVICE)
    with pytest.raises(newt.CompileError, match="scalar pointer"):
        k[(1,)](x, y, BLOCK=64)


def test_pointer_rebase_rejected():
    @newt.jit
    def k(a_ptr, b_ptr, out_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        p = a_ptr + offs
        for i in range(2):
            v = nl.load(p)
            nl.store(out_ptr + i * BLOCK + offs, v)
            p = b_ptr + offs

    a = torch.ones(64, device=DEVICE)
    b = torch.full((64,), 2.0, device=DEVICE)
    out = torch.empty(128, device=DEVICE)
    with pytest.raises(newt.CompileError, match="base"):
        k[(1,)](a, b, out, BLOCK=64)


def test_int_storage_float_update_rejected():
    @newt.jit
    def k(y_ptr, ITERS: nl.constexpr):
        s = 0
        for i in range(ITERS):
            s = s + 0.5
        nl.store(y_ptr, s)

    y = torch.empty(1, device=DEVICE)
    with pytest.raises(newt.CompileError, match="float"):
        k[(1,)](y, ITERS=3)


def test_scalar_adopts_block_dtype():
    @newt.jit
    def k(a_ptr, o_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        nl.store(o_ptr + offs, nl.load(a_ptr + offs) + 200)

    a = torch.full((64,), 200, device=DEVICE, dtype=torch.uint8)
    o = torch.empty(64, device=DEVICE, dtype=torch.uint8)
    k[(1,)](a, o, BLOCK=64)
    assert o[0].item() == 144  # uint8 wraparound, matching Triton

    @newt.jit
    def kh(a_ptr, o_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        nl.store(o_ptr + offs, nl.load(a_ptr + offs) * 0.1)

    ah = torch.full((64,), 3.0, device=DEVICE, dtype=torch.float16)
    oh = torch.empty(64, device=DEVICE, dtype=torch.float32)
    kh[(1,)](ah, oh, BLOCK=64)
    # fp16 math (0.299804688), not fp32 (0.300000012)
    assert abs(oh[0].item() - 0.299804688) < 1e-6


def test_broadcast_to_uniform_scalar():
    @newt.jit
    def k(x_ptr, y_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        s = nl.sum(nl.load(x_ptr + offs))
        nl.store(y_ptr + offs, nl.broadcast_to(s, (BLOCK,)))

    x = torch.ones(64, device=DEVICE)
    y = torch.empty(64, device=DEVICE)
    k[(1,)](x, y, BLOCK=64)
    assert torch.all(y == 64.0)


def test_load_other_from_load():
    @newt.jit
    def k(p_ptr, q_ptr, y_ptr, n, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        fallback = nl.load(q_ptr + offs)
        m = offs < n
        v = nl.load(p_ptr + offs, mask=m, other=fallback)
        nl.store(y_ptr + offs, v)

    p = torch.ones(64, device=DEVICE)
    q = torch.full((64,), 5.0, device=DEVICE)
    y = torch.empty(64, device=DEVICE)
    k[(1,)](p, q, y, 4, BLOCK=64)
    assert torch.all(y[:4] == 1.0) and torch.all(y[4:] == 5.0)


def test_unknown_kwargs_rejected():
    @newt.jit
    def k(x_ptr, BLOCK: nl.constexpr):
        nl.store(x_ptr + nl.arange(0, BLOCK), 1.0)

    x = torch.zeros(64, device=DEVICE)
    with pytest.raises(TypeError, match="unknown"):
        k[(1,)](x, BLOCK=64, BLOK=128)
    with pytest.raises(TypeError, match="arguments"):
        k[(1,)](x, 64, 999)


def test_uint32_tensors():
    @newt.jit
    def k(x_ptr, y_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        nl.store(y_ptr + offs, nl.load(x_ptr + offs) + 1)

    x = torch.zeros(64, device=DEVICE, dtype=torch.uint32)
    y = torch.empty_like(x)
    k[(1,)](x, y, BLOCK=64)
    assert int(y[0]) == 1


def test_constexpr_host_arithmetic():
    c = nl.constexpr(64)
    assert (c * 2).value == 128
    assert int(c) == 64
    assert c == 64


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
