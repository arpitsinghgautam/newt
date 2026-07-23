"""CPU-only compiler tests: Python AST -> CUDA C++ without touching a GPU.

These validate the *structure* of generated code (layouts, vector fast
paths, reductions, tensor-core staging, pipelining, smem accounting) so CI
runners without CUDA still cover the compiler. Numerical correctness lives
in the GPU suites.
"""

import ast
import inspect
import textwrap

import pytest

import newt.language as nl
from newt.compiler import codegen
from newt.compiler import types as tp

pytestmark = pytest.mark.cpu


def compile_source(fn, arg_types, constexprs, num_warps=4, num_stages=3):
    """Run the code generator directly (no NVRTC, no driver, no GPU)."""
    src = textwrap.dedent(inspect.getsource(fn))
    fndef = ast.parse(src).body[0]
    params = [(n, codegen.Value(codegen.UNIFORM, t, (), n))
              for n, t in arg_types.items()]
    return codegen.compile_fn(fndef, fn.__globals__, params, constexprs,
                              num_warps, "k", num_stages=num_stages)


F32P = tp.pointer_type(tp.float32)
F16P = tp.pointer_type(tp.float16)


def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: nl.constexpr):
    pid = nl.program_id(0)
    offs = pid * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask)
    nl.store(out_ptr + offs, x + y, mask=mask)


def test_vector_fast_path_and_layout():
    src, smem = compile_source(
        add_kernel,
        {"x_ptr": F32P, "y_ptr": F32P, "out_ptr": F32P, "n": tp.int32},
        {"BLOCK": 1024})
    assert "__launch_bounds__(128)" in src
    assert "_nv<float, 4, 16>" in src        # 16-byte vectorized fast path
    assert "blockIdx.x" in src
    assert "__restrict__" in src
    assert smem == 0                          # pure elementwise: no smem


def softmax_kernel(x_ptr, out_ptr, N, stride, BLOCK_N: nl.constexpr):
    row = nl.program_id(0)
    cols = nl.arange(0, BLOCK_N)
    mask = cols < N
    x = nl.load(x_ptr + row * stride + cols, mask=mask, other=float("-inf"))
    x = x - nl.max(x, axis=0)
    num = nl.exp(x)
    den = nl.sum(num, axis=0)
    nl.store(out_ptr + row * stride + cols, num / den, mask=mask)


def test_reduction_lowering():
    src, smem = compile_source(
        softmax_kernel,
        {"x_ptr": F32P, "out_ptr": F32P, "N": tp.int32, "stride": tp.int32},
        {"BLOCK_N": 2048}, num_warps=8)
    assert "__shfl_xor_sync" in src           # warp butterfly reduction
    assert src.count("__shfl_xor_sync") >= 2  # in-warp + cross-warp stages
    assert "expf(" in src
    assert smem > 0                           # cross-warp scratch


def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: nl.constexpr, BLOCK_N: nl.constexpr,
                  BLOCK_K: nl.constexpr):
    pid_m = nl.program_id(0)
    pid_n = nl.program_id(1)
    offs_m = pid_m * BLOCK_M + nl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + nl.arange(0, BLOCK_N)
    offs_k = nl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = nl.zeros((BLOCK_M, BLOCK_N), dtype=nl.float32)
    for k in range(K // BLOCK_K):
        a = nl.load(a_ptrs)
        b = nl.load(b_ptrs)
        acc = nl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    nl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


MM_TYPES = {"a_ptr": F16P, "b_ptr": F16P, "c_ptr": F16P,
            "M": tp.int32, "N": tp.int32, "K": tp.int32,
            "sam": tp.int32, "sak": tp.int32, "sbk": tp.int32,
            "sbn": tp.int32, "scm": tp.int32, "scn": tp.int32}
MM_CONST = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}


def test_pipelined_dot_structure():
    src, smem = compile_source(matmul_kernel, MM_TYPES, MM_CONST)
    assert "_mma_f16(" in src                 # raw mma.sync PTX core
    assert "_ldm4(" in src and "_ldm2t(" in src   # ldmatrix fragment loads
    assert "_nsw(" in src                     # XOR-swizzled smem
    assert "__pipeline_memcpy_async" in src   # cp.async staging
    assert "#define _NRB0" in src             # dedicated ring region
    assert "_dpp0" in src and "_dpb0" in src  # pipeline state
    assert "__pipeline_wait_prior(1)" in src  # S=3: wait with S-2 newer commits
    # ring: 3 slots of (A + B tile), unpadded (swizzle replaces padding)
    esz = 2
    bytesA = (64 * 32 * esz + 15) // 16 * 16
    bytesB = (32 * 64 * esz + 15) // 16 * 16
    ring = 3 * (bytesA + bytesB)
    assert smem >= ring
    base = int(src.split("#define _NRB0 ")[1].split("\n")[0])
    assert base + ring == smem                # ring sits after the scratch arena


def test_pipeline_env_fallbacks(monkeypatch):
    monkeypatch.setenv("NEWT_PIPELINE_DOT", "0")
    src, _ = compile_source(matmul_kernel, MM_TYPES, MM_CONST)
    assert "_NRB0" not in src                 # chunked path, no ring
    assert "__pipeline_memcpy_async" in src
    monkeypatch.setenv("NEWT_ASYNC_DOT", "0")
    src, _ = compile_source(matmul_kernel, MM_TYPES, MM_CONST)
    assert "__pipeline_memcpy_async" not in src   # fully synchronous staging
    assert "_mma_f16(" in src
    monkeypatch.setenv("NEWT_MMA", "wmma")
    src, _ = compile_source(matmul_kernel, MM_TYPES, MM_CONST)
    assert "wmma::mma_sync" in src            # WMMA fallback still available


def test_constexpr_branch_pruning():
    def k(x_ptr, o_ptr, MODE: nl.constexpr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        x = nl.load(x_ptr + offs)
        if MODE == "double":
            r = x * 2
        else:
            r = nl.exp(x)
        nl.store(o_ptr + offs, r)

    src, _ = compile_source(k, {"x_ptr": F32P, "o_ptr": F32P},
                            {"MODE": "double", "BLOCK": 64})
    assert "expf" not in src                  # untaken branch pruned
    src, _ = compile_source(k, {"x_ptr": F32P, "o_ptr": F32P},
                            {"MODE": "exp", "BLOCK": 64})
    assert "expf" in src


def test_tf32_dot_path():
    types = dict(MM_TYPES)
    for p in ("a_ptr", "b_ptr", "c_ptr"):
        types[p] = F32P
    src, _ = compile_source(matmul_kernel, types, MM_CONST)
    assert "wmma::precision::tf32" in src
    assert "__float_to_tf32" in src


def test_compile_errors_are_clean():
    def bad(x_ptr, BLOCK: nl.constexpr):
        offs = nl.arange(0, BLOCK)
        x = nl.load(x_ptr + offs)
        if x > 0:  # block condition: rejected
            nl.store(x_ptr, 1.0)

    with pytest.raises(codegen.CompileError, match="scalar"):
        compile_source(bad, {"x_ptr": F32P}, {"BLOCK": 64})
