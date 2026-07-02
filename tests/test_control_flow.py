"""Control flow, constexpr, autotune, and runtime-feature tests."""

import pytest
import torch

import newt
import newt.language as nl

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# loops
# ---------------------------------------------------------------------------

def test_runtime_loop_accumulator():
    @newt.jit
    def k(x_ptr, o_ptr, n_iter, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        acc = nl.zeros((BLOCK,), dtype=nl.float32)
        for i in range(n_iter):
            acc = acc + nl.load(x_ptr + i * BLOCK + offs)
        nl.store(o_ptr + offs, acc)

    iters, B = 7, 256
    x = torch.randn(iters * B, device=DEVICE)
    o = torch.empty(B, device=DEVICE)
    k[(1,)](x, o, iters, BLOCK=B)
    assert torch.allclose(o, x.view(iters, B).sum(0), rtol=1e-5, atol=1e-5)


def test_constexpr_loop_unrolled():
    @newt.jit
    def k(o_ptr, N: nl.constexpr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        acc = nl.zeros((BLOCK,), dtype=nl.int32)
        for i in range(0, N, 2):
            acc = acc + i
        nl.store(o_ptr + offs, acc)

    o = torch.empty(64, device=DEVICE, dtype=torch.int32)
    k[(1,)](o, N=10, BLOCK=64)
    assert torch.all(o == sum(range(0, 10, 2)))


def test_scalar_loop_carried():
    @newt.jit
    def k(o_ptr, n, BLOCK: nl.constexpr):
        s = 0
        for i in range(n):
            s = s + i * i
        nl.store(o_ptr + nl.arange(0, BLOCK), nl.zeros((BLOCK,), dtype=nl.int32) + s)

    o = torch.empty(32, device=DEVICE, dtype=torch.int32)
    k[(1,)](o, 10, BLOCK=32)
    assert torch.all(o == sum(i * i for i in range(10)))


def test_pointer_walk():
    @newt.jit
    def k(x_ptr, o_ptr, n_tiles, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        p = x_ptr + offs
        acc = nl.zeros((BLOCK,), dtype=nl.float32)
        for i in range(n_tiles):
            acc = acc + nl.load(p)
            p += BLOCK
        nl.store(o_ptr + offs, acc)

    tiles, B = 5, 512
    x = torch.randn(tiles * B, device=DEVICE)
    o = torch.empty(B, device=DEVICE)
    k[(1,)](x, o, tiles, BLOCK=B)
    assert torch.allclose(o, x.view(tiles, B).sum(0), rtol=1e-5, atol=1e-5)


def test_nested_loops_and_break_continue():
    @newt.jit
    def k(o_ptr, BLOCK: nl.constexpr):
        s = 0
        for i in range(10):
            if i == 7:
                break
            if i % 2 == 1:
                continue
            for j in range(3):
                s = s + i + j
        nl.store(o_ptr + nl.arange(0, BLOCK), nl.zeros((BLOCK,), dtype=nl.int32) + s)

    expected = 0
    for i in range(10):
        if i == 7:
            break
        if i % 2 == 1:
            continue
        for j in range(3):
            expected += i + j
    o = torch.empty(32, device=DEVICE, dtype=torch.int32)
    k[(1,)](o, BLOCK=32)
    assert torch.all(o == expected)


def test_while_loop():
    @newt.jit
    def k(o_ptr, n, BLOCK: nl.constexpr):
        v = 1
        while v < n:
            v = v * 2
        nl.store(o_ptr + nl.arange(0, BLOCK), nl.zeros((BLOCK,), dtype=nl.int32) + v)

    o = torch.empty(32, device=DEVICE, dtype=torch.int32)
    k[(1,)](o, 100, BLOCK=32)
    assert torch.all(o == 128)


# ---------------------------------------------------------------------------
# if / constexpr pruning
# ---------------------------------------------------------------------------

def test_runtime_if():
    @newt.jit
    def k(x_ptr, o_ptr, flag, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        x = nl.load(x_ptr + offs)
        y = nl.zeros((BLOCK,), dtype=nl.float32)
        if flag == 1:
            y = x * 2
        else:
            y = x * 3
        nl.store(o_ptr + offs, y)

    x = torch.randn(128, device=DEVICE)
    o = torch.empty_like(x)
    k[(1,)](x, o, 1, BLOCK=128)
    assert torch.equal(o, x * 2)
    k[(1,)](x, o, 0, BLOCK=128)
    assert torch.equal(o, x * 3)


def test_constexpr_branch_pruning():
    @newt.jit
    def k(x_ptr, o_ptr, MODE: nl.constexpr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        x = nl.load(x_ptr + offs)
        if MODE == "double":
            r = x * 2
        else:
            r = nl.exp(x)  # pruned entirely when MODE == "double"
        nl.store(o_ptr + offs, r)

    x = torch.randn(64, device=DEVICE)
    o = torch.empty_like(x)
    k[(1,)](x, o, MODE="double", BLOCK=64)
    assert torch.equal(o, x * 2)
    k[(1,)](x, o, MODE="exp", BLOCK=64)
    assert torch.allclose(o, torch.exp(x), rtol=1e-6, atol=1e-6)


def test_static_assert_fires():
    @newt.jit
    def k(o_ptr, BLOCK: nl.constexpr):
        nl.static_assert(BLOCK >= 128, "BLOCK too small")
        nl.store(o_ptr + nl.arange(0, BLOCK), 1.0)

    o = torch.empty(64, device=DEVICE)
    with pytest.raises(newt.CompileError, match="BLOCK too small"):
        k[(1,)](o, BLOCK=64)


def test_block_condition_rejected():
    @newt.jit
    def k(x_ptr, o_ptr, BLOCK: nl.constexpr):
        x = nl.load(x_ptr + nl.arange(0, BLOCK))
        if x > 0:  # blocks can't be branch conditions
            nl.store(o_ptr, 1.0)

    x = torch.randn(64, device=DEVICE)
    o = torch.empty(1, device=DEVICE)
    with pytest.raises(newt.CompileError, match="scalar"):
        k[(1,)](x, o, BLOCK=64)


# ---------------------------------------------------------------------------
# grids / program_id
# ---------------------------------------------------------------------------

def test_3d_grid():
    @newt.jit
    def k(o_ptr, GX: nl.constexpr, GY: nl.constexpr):
        px = nl.program_id(0)
        py = nl.program_id(1)
        pz = nl.program_id(2)
        idx = (pz * GY + py) * GX + px
        nl.store(o_ptr + idx, idx.to(nl.float32))

    GX, GY, GZ = 4, 3, 2
    o = torch.empty(GX * GY * GZ, device=DEVICE)
    k[(GX, GY, GZ)](o, GX=GX, GY=GY)
    assert torch.equal(o, torch.arange(GX * GY * GZ, device=DEVICE).float())


def test_num_programs():
    @newt.jit
    def k(o_ptr):
        pid = nl.program_id(0)
        nl.store(o_ptr + pid, nl.num_programs(0).to(nl.float32))

    o = torch.empty(7, device=DEVICE)
    k[(7,)](o)
    assert torch.all(o == 7)


# ---------------------------------------------------------------------------
# autotune / heuristics / cache
# ---------------------------------------------------------------------------

def test_autotune():
    @newt.autotune(
        configs=[
            newt.Config({"BLOCK": 256}, num_warps=2),
            newt.Config({"BLOCK": 1024}, num_warps=4),
            newt.Config({"BLOCK": 4096}, num_warps=8),
        ],
        key=["n"],
    )
    @newt.jit
    def k(x_ptr, o_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        nl.store(o_ptr + offs, nl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    n = 1 << 20
    x = torch.randn(n, device=DEVICE)
    o = torch.empty_like(x)
    grid = lambda meta: (newt.cdiv(n, meta["BLOCK"]),)
    k[grid](x, o, n)
    assert torch.equal(o, x + 1)
    assert k.best_config is not None


def test_heuristics():
    @newt.heuristics({"BLOCK": lambda args: newt.next_power_of_2(min(args["n"], 1024))})
    @newt.jit
    def k(x_ptr, o_ptr, n, BLOCK: nl.constexpr):
        offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
        mask = offs < n
        nl.store(o_ptr + offs, nl.load(x_ptr + offs, mask=mask) * 5, mask=mask)

    for n in (100, 5000):
        x = torch.randn(n, device=DEVICE)
        o = torch.empty_like(x)
        k[lambda meta: (newt.cdiv(n, meta["BLOCK"]),)](x, o, n)
        assert torch.equal(o, x * 5)


def test_specialization_cache():
    @newt.jit
    def k(x_ptr, o_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        nl.store(o_ptr + offs, nl.load(x_ptr + offs) + 1)

    for dtype in (torch.float32, torch.float16):
        for block in (64, 128):
            x = torch.randn(block, device=DEVICE).to(dtype)
            o = torch.empty_like(x)
            k[(1,)](x, o, BLOCK=block)
            assert torch.allclose(o.float(), x.float() + 1, rtol=1e-3, atol=1e-3)
    assert len(k.cache) == 4  # 2 dtypes x 2 block sizes


def test_int64_specialization():
    @newt.jit
    def k(o_ptr, big, BLOCK: nl.constexpr):
        nl.store(o_ptr + nl.arange(0, BLOCK),
                 nl.zeros((BLOCK,), dtype=nl.float32) + (big / 4294967296))

    o = torch.empty(32, device=DEVICE)
    k[(1,)](o, 1 << 40, BLOCK=32)  # doesn't fit int32 -> int64 param
    assert torch.all(o == float((1 << 40) // 4294967296))


# ---------------------------------------------------------------------------
# edge shapes
# ---------------------------------------------------------------------------

def test_block_smaller_than_warp_count():
    @newt.jit
    def k(x_ptr, o_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        nl.store(o_ptr + offs, nl.load(x_ptr + offs) + 1)

    x = torch.randn(16, device=DEVICE)
    o = torch.empty_like(x)
    k[(1,)](x, o, BLOCK=16, num_warps=4)  # 128 threads, 16 elements
    assert torch.equal(o, x + 1)


def test_n_equals_one():
    @newt.jit
    def k(x_ptr, o_ptr, n, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        mask = offs < n
        nl.store(o_ptr + offs, nl.load(x_ptr + offs, mask=mask) * 3, mask=mask)

    x = torch.randn(1, device=DEVICE)
    o = torch.empty_like(x)
    k[(1,)](x, o, 1, BLOCK=64)
    assert torch.equal(o, x * 3)


def test_tuple_unpack():
    @newt.jit
    def k(o_ptr, BLOCK: nl.constexpr):
        a, b = nl.arange(0, BLOCK), nl.arange(0, BLOCK)
        nl.store(o_ptr + a, (a + b).to(nl.float32))

    o = torch.empty(64, device=DEVICE)
    k[(1,)](o, BLOCK=64)
    assert torch.equal(o, torch.arange(64, device=DEVICE).float() * 2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
