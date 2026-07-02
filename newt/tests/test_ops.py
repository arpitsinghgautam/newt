"""Comprehensive op correctness tests: every nl.* op vs a torch reference."""

import math

import pytest
import torch

import newt
import newt.language as nl

DEVICE = "cuda"


def make(shape, dtype=torch.float32, positive=False, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(shape, device=DEVICE, generator=g).to(dtype)
    if positive:
        x = x.abs() + 0.5
    return x


# ---------------------------------------------------------------------------
# elementwise binary ops
# ---------------------------------------------------------------------------

@newt.jit
def _binop_kernel(x_ptr, y_ptr, o_ptr, n, OP: nl.constexpr, BLOCK: nl.constexpr):
    offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask, other=1.0)
    if OP == "add":
        r = x + y
    if OP == "sub":
        r = x - y
    if OP == "mul":
        r = x * y
    if OP == "div":
        r = x / y
    if OP == "pow2":
        r = x ** 2
    if OP == "max":
        r = nl.maximum(x, y)
    if OP == "min":
        r = nl.minimum(x, y)
    if OP == "fma":
        r = nl.fma(x, y, x)
    if OP == "where":
        r = nl.where(x > 0, x, y)
    nl.store(o_ptr + offs, r, mask=mask)


BINOPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "pow2": lambda x, y: x * x,
    "max": torch.maximum,
    "min": torch.minimum,
    "fma": lambda x, y: x * y + x,
    "where": lambda x, y: torch.where(x > 0, x, y),
}


@pytest.mark.parametrize("op", list(BINOPS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_binops(op, dtype):
    n = 4321
    x = make(n, dtype)
    y = make(n, dtype, positive=True, seed=1)
    o = torch.empty_like(x)
    _binop_kernel[(newt.cdiv(n, 1024),)](x, y, o, n, OP=op, BLOCK=1024)
    ref = BINOPS[op](x, y)
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert torch.allclose(o.float(), ref.float(), rtol=tol, atol=tol), op


# ---------------------------------------------------------------------------
# int ops
# ---------------------------------------------------------------------------

@newt.jit
def _intop_kernel(x_ptr, y_ptr, o_ptr, n, OP: nl.constexpr, BLOCK: nl.constexpr):
    offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask, other=1)
    if OP == "and":
        r = x & y
    if OP == "or":
        r = x | y
    if OP == "xor":
        r = x ^ y
    if OP == "shl":
        r = x << 1
    if OP == "mod":
        r = x % y
    if OP == "floordiv":
        r = x // y
    nl.store(o_ptr + offs, r, mask=mask)


@pytest.mark.parametrize("op,ref", [
    ("and", lambda x, y: x & y),
    ("or", lambda x, y: x | y),
    ("xor", lambda x, y: x ^ y),
    ("shl", lambda x, y: x << 1),
    ("mod", lambda x, y: x % y),        # positive operands: C == python
    ("floordiv", lambda x, y: x // y),  # positive operands: C == python
])
def test_int_ops(op, ref):
    n = 2048
    x = torch.randint(0, 1000, (n,), device=DEVICE, dtype=torch.int32)
    y = torch.randint(1, 50, (n,), device=DEVICE, dtype=torch.int32)
    o = torch.empty_like(x)
    _intop_kernel[(newt.cdiv(n, 1024),)](x, y, o, n, OP=op, BLOCK=1024)
    assert torch.equal(o, ref(x, y)), op


# ---------------------------------------------------------------------------
# unary math
# ---------------------------------------------------------------------------

@newt.jit
def _unary_kernel(x_ptr, o_ptr, n, OP: nl.constexpr, BLOCK: nl.constexpr):
    offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask, other=1.0)
    if OP == "exp":
        r = nl.exp(x)
    if OP == "exp2":
        r = nl.exp2(x)
    if OP == "log":
        r = nl.log(x)
    if OP == "log2":
        r = nl.log2(x)
    if OP == "sqrt":
        r = nl.sqrt(x)
    if OP == "rsqrt":
        r = nl.rsqrt(x)
    if OP == "sin":
        r = nl.sin(x)
    if OP == "cos":
        r = nl.cos(x)
    if OP == "tanh":
        r = nl.tanh(x)
    if OP == "erf":
        r = nl.erf(x)
    if OP == "sigmoid":
        r = nl.sigmoid(x)
    if OP == "abs":
        r = nl.abs(x)
    if OP == "floor":
        r = nl.floor(x)
    if OP == "ceil":
        r = nl.ceil(x)
    if OP == "neg":
        r = -x
    nl.store(o_ptr + offs, r, mask=mask)


UNARY = {
    "exp": torch.exp, "exp2": torch.exp2, "log": torch.log, "log2": torch.log2,
    "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "sin": torch.sin, "cos": torch.cos,
    "tanh": torch.tanh, "erf": torch.erf, "sigmoid": torch.sigmoid,
    "abs": torch.abs, "floor": torch.floor, "ceil": torch.ceil,
    "neg": torch.neg,
}


@pytest.mark.parametrize("op", list(UNARY))
def test_unary(op):
    n = 3000
    positive = op in ("log", "log2", "sqrt", "rsqrt")
    x = make(n, positive=positive)
    o = torch.empty_like(x)
    _unary_kernel[(newt.cdiv(n, 1024),)](x, o, n, OP=op, BLOCK=1024)
    assert torch.allclose(o, UNARY[op](x), rtol=1e-5, atol=1e-5), op


# ---------------------------------------------------------------------------
# casts
# ---------------------------------------------------------------------------

@newt.jit
def _cast_kernel(x_ptr, o_ptr, n, DT: nl.constexpr, BLOCK: nl.constexpr):
    offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    nl.store(o_ptr + offs, x.to(DT), mask=mask)


@pytest.mark.parametrize("src,dst,ndt", [
    (torch.float32, torch.float16, nl.float16),
    (torch.float16, torch.float32, nl.float32),
    (torch.float32, torch.bfloat16, nl.bfloat16),
    (torch.float32, torch.int32, nl.int32),
    (torch.int32, torch.float32, nl.float32),
    (torch.int64, torch.int32, nl.int32),
])
def test_cast(src, dst, ndt):
    n = 1024
    x = (make(n) * 10).to(src)
    o = torch.empty(n, device=DEVICE, dtype=dst)
    _cast_kernel[(1,)](x, o, n, DT=ndt, BLOCK=1024)
    ref = x.to(dst)
    if dst.is_floating_point:
        assert torch.allclose(o.float(), ref.float(), rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(o, ref)


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------

@newt.jit
def _reduce2d_kernel(x_ptr, o_ptr, M, N, sx, OP: nl.constexpr, AXIS: nl.constexpr,
                     BM: nl.constexpr, BN: nl.constexpr):
    rm = nl.arange(0, BM)
    rn = nl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if OP == "sum":
        x = nl.load(x_ptr + rm[:, None] * sx + rn[None, :], mask=mask, other=0.0)
        r = nl.sum(x, axis=AXIS)
    if OP == "max":
        x = nl.load(x_ptr + rm[:, None] * sx + rn[None, :], mask=mask,
                    other=float("-inf"))
        r = nl.max(x, axis=AXIS)
    if OP == "min":
        x = nl.load(x_ptr + rm[:, None] * sx + rn[None, :], mask=mask,
                    other=float("inf"))
        r = nl.min(x, axis=AXIS)
    offs_o = rn if AXIS == 0 else rm
    omask = (offs_o < N) if AXIS == 0 else (offs_o < M)
    nl.store(o_ptr + offs_o, r, mask=omask)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("mn", [(32, 64), (17, 100), (128, 128)])
def test_reduce_axis(op, axis, mn):
    M, N = mn
    x = make((M, N))
    olen = N if axis == 0 else M
    o = torch.empty(olen, device=DEVICE)
    BM, BN = newt.next_power_of_2(M), newt.next_power_of_2(N)
    _reduce2d_kernel[(1,)](x, o, M, N, x.stride(0), OP=op, AXIS=axis, BM=BM, BN=BN)
    ref = {"sum": torch.sum, "max": lambda t, dim: t.amax(dim),
           "min": lambda t, dim: t.amin(dim)}[op](x, dim=axis) if op == "sum" else \
        (x.amax(axis) if op == "max" else x.amin(axis))
    assert torch.allclose(o, ref, rtol=1e-4, atol=1e-4), (op, axis, mn)


@newt.jit
def _reduce_full_kernel(x_ptr, o_ptr, n, OP: nl.constexpr, BLOCK: nl.constexpr):
    offs = nl.arange(0, BLOCK)
    mask = offs < n
    if OP == "sum":
        x = nl.load(x_ptr + offs, mask=mask, other=0.0)
        r = nl.sum(x)
    if OP == "max":
        x = nl.load(x_ptr + offs, mask=mask, other=float("-inf"))
        r = nl.max(x)
    nl.store(o_ptr, r)


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("n", [32, 100, 1024, 5000])
@pytest.mark.parametrize("warps", [1, 4, 8])
def test_reduce_full(op, n, warps):
    x = make(n)
    o = torch.empty(1, device=DEVICE)
    _reduce_full_kernel[(1,)](x, o, n, OP=op, BLOCK=newt.next_power_of_2(n),
                              num_warps=warps)
    ref = x.sum() if op == "sum" else x.max()
    assert torch.allclose(o[0], ref, rtol=1e-4, atol=1e-4)


def test_reduce_fp16_promotes():
    @newt.jit
    def k(x_ptr, o_ptr, BLOCK: nl.constexpr):
        x = nl.load(x_ptr + nl.arange(0, BLOCK))
        nl.store(o_ptr, nl.sum(x))

    x = torch.ones(4096, device=DEVICE, dtype=torch.float16)
    o = torch.empty(1, device=DEVICE)  # fp32 result
    k[(1,)](x, o, BLOCK=4096)
    assert o.item() == 4096.0  # fp16 accumulation would lose precision at 2048


# ---------------------------------------------------------------------------
# broadcasting / shape ops
# ---------------------------------------------------------------------------

def test_broadcast_2d():
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, BM: nl.constexpr, BN: nl.constexpr):
        a = nl.load(a_ptr + nl.arange(0, BM))
        b = nl.load(b_ptr + nl.arange(0, BN))
        r = a[:, None] * 100 + b[None, :]
        offs = nl.arange(0, BM)[:, None] * BN + nl.arange(0, BN)[None, :]
        nl.store(o_ptr + offs, r)

    M, N = 32, 64
    a = make(M)
    b = make(N, seed=1)
    o = torch.empty(M, N, device=DEVICE)
    k[(1,)](a, b, o, BM=M, BN=N)
    assert torch.allclose(o, a[:, None] * 100 + b[None, :], rtol=1e-5, atol=1e-5)


def test_broadcast_to_and_reshape():
    @newt.jit
    def k(a_ptr, o_ptr, BM: nl.constexpr, BN: nl.constexpr):
        a = nl.load(a_ptr + nl.arange(0, BM))
        r = nl.broadcast_to(a[:, None], (BM, BN))
        r = nl.reshape(r, (BM * BN,))
        nl.store(o_ptr + nl.arange(0, BM * BN), r)

    M, N = 16, 32
    a = make(M)
    o = torch.empty(M * N, device=DEVICE)
    k[(1,)](a, o, BM=M, BN=N)
    assert torch.equal(o.view(M, N), a[:, None].expand(M, N))


def test_trans():
    @newt.jit
    def k(x_ptr, o_ptr, BM: nl.constexpr, BN: nl.constexpr):
        offs = nl.arange(0, BM)[:, None] * BN + nl.arange(0, BN)[None, :]
        x = nl.load(x_ptr + offs)
        t = nl.trans(x)
        offs_t = nl.arange(0, BN)[:, None] * BM + nl.arange(0, BM)[None, :]
        nl.store(o_ptr + offs_t, t)

    M, N = 32, 64
    x = make((M, N))
    o = torch.empty(N, M, device=DEVICE)
    k[(1,)](x, o, BM=M, BN=N)
    assert torch.equal(o, x.t().contiguous())


# ---------------------------------------------------------------------------
# loads/stores: strided, gather, uniform
# ---------------------------------------------------------------------------

def test_strided_noncontiguous():
    @newt.jit
    def k(x_ptr, o_ptr, n, sx, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        x = nl.load(x_ptr + offs * sx, mask=mask)  # stride-2: vector fast path must bail
        nl.store(o_ptr + offs, x * 2, mask=mask)

    base = make(2000)
    x = base[::2]  # non-contiguous view
    n = x.shape[0]
    o = torch.empty(n, device=DEVICE)
    k[(newt.cdiv(n, 256),)](base, o, n, 2, BLOCK=256)
    assert torch.equal(o, x * 2)


def test_gather_indices():
    @newt.jit
    def k(x_ptr, idx_ptr, o_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        idx = nl.load(idx_ptr + offs, mask=mask)
        x = nl.load(x_ptr + idx, mask=mask)
        nl.store(o_ptr + offs, x, mask=mask)

    n = 4096
    x = make(n)
    idx = torch.randperm(n, device=DEVICE).to(torch.int32)
    o = torch.empty(n, device=DEVICE)
    k[(newt.cdiv(n, 1024),)](x, idx, o, n, BLOCK=1024)
    assert torch.equal(o, x[idx.long()])


def test_scalar_pointer_load():
    @newt.jit
    def k(x_ptr, o_ptr, BLOCK: nl.constexpr):
        s = nl.load(x_ptr)  # uniform scalar load
        offs = nl.arange(0, BLOCK)
        nl.store(o_ptr + offs, nl.zeros((BLOCK,), dtype=nl.float32) + s)

    x = torch.tensor([42.0], device=DEVICE)
    o = torch.empty(128, device=DEVICE)
    k[(1,)](x, o, BLOCK=128)
    assert torch.all(o == 42.0)


# ---------------------------------------------------------------------------
# atomics
# ---------------------------------------------------------------------------

def test_atomic_add_histogram():
    @newt.jit
    def k(idx_ptr, hist_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        idx = nl.load(idx_ptr + offs, mask=mask)
        nl.atomic_add(hist_ptr + idx, 1.0, mask=mask)

    n, bins = 100_000, 64
    idx = torch.randint(0, bins, (n,), device=DEVICE, dtype=torch.int32)
    hist = torch.zeros(bins, device=DEVICE)
    k[(newt.cdiv(n, 1024),)](idx, hist, n, BLOCK=1024)
    ref = torch.bincount(idx.long(), minlength=bins).float()
    assert torch.equal(hist, ref)


def test_atomic_max():
    @newt.jit
    def k(x_ptr, o_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        x = nl.load(x_ptr + offs, mask=mask, other=0)
        nl.atomic_max(o_ptr, nl.max(x))

    n = 10_000
    x = torch.randint(0, 1_000_000, (n,), device=DEVICE, dtype=torch.int32)
    o = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    k[(newt.cdiv(n, 1024),)](x, o, n, BLOCK=1024)
    assert o.item() == x.max().item()


# ---------------------------------------------------------------------------
# dot: shapes / dtypes / chained
# ---------------------------------------------------------------------------

@newt.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, BM: nl.constexpr, BN: nl.constexpr,
                BK: nl.constexpr):
    ra, rk, rb = nl.arange(0, BM), nl.arange(0, BK), nl.arange(0, BN)
    a = nl.load(a_ptr + ra[:, None] * BK + rk[None, :])
    b = nl.load(b_ptr + rk[:, None] * BN + rb[None, :])
    c = nl.dot(a, b)
    nl.store(c_ptr + ra[:, None] * BN + rb[None, :], c)


@pytest.mark.parametrize("dtype,tol", [
    (torch.float16, 2e-2), (torch.bfloat16, 3e-2), (torch.float32, 2e-2),
])
@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 128), (64, 32, 16)])
@pytest.mark.parametrize("warps", [2, 4])
def test_dot(dtype, tol, shape, warps):
    BM, BN, BK = shape
    a = make((BM, BK), dtype) * 0.3
    b = make((BK, BN), dtype, seed=1) * 0.3
    c = torch.empty(BM, BN, device=DEVICE, dtype=torch.float32)
    _dot_kernel[(1,)](a, b, c, BM=BM, BN=BN, BK=BK, num_warps=warps)
    ref = a.float() @ b.float()
    assert torch.allclose(c, ref, rtol=tol, atol=tol)


def test_dot_epilogue_ops():
    """dot result flows through elementwise + reduction + store."""
    @newt.jit
    def k(a_ptr, b_ptr, o_ptr, BM: nl.constexpr, BN: nl.constexpr, BK: nl.constexpr):
        ra, rk, rb = nl.arange(0, BM), nl.arange(0, BK), nl.arange(0, BN)
        a = nl.load(a_ptr + ra[:, None] * BK + rk[None, :])
        b = nl.load(b_ptr + rk[:, None] * BN + rb[None, :])
        c = nl.dot(a, b) * 2.0 + 1.0
        nl.store(o_ptr + ra, nl.sum(c, axis=1))

    BM, BN, BK = 32, 32, 32
    a = make((BM, BK), torch.float16) * 0.3
    b = make((BK, BN), torch.float16, seed=1) * 0.3
    o = torch.empty(BM, device=DEVICE)
    k[(1,)](a, b, o, BM=BM, BN=BN, BK=BK)
    ref = (a.float() @ b.float() * 2 + 1).sum(1)
    assert torch.allclose(o, ref, rtol=2e-2, atol=2e-1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
