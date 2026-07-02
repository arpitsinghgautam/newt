# newt 🦎

A **mini-Triton**: the [Triton](https://github.com/triton-lang/triton) GPU
programming model - block-level kernels written in Python, JIT-compiled to
real GPU machine code - reimplemented from scratch in ~3,000 lines of
readable Python, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

> *Why "newt"? **Triton** was the original genus name for newts
> (Laurenti, 1768). A newt is literally a small triton.*

```python
import torch
import newt
import newt.language as nl

@newt.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: nl.constexpr):
    pid = nl.program_id(0)
    offs = pid * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask)
    nl.store(out_ptr + offs, x + y, mask=mask)

x, y = torch.randn(2, 1_000_000, device="cuda")
out = torch.empty_like(x)
add_kernel[lambda meta: (newt.cdiv(1_000_000, meta["BLOCK"]),)](
    x, y, out, 1_000_000, BLOCK=1024)
```

If you know Triton, you know newt: replace `tl` with `nl` and it usually just
runs - same `@jit`/grid launch protocol, same `constexpr` specialization,
same `@autotune`/`@heuristics` decorators, same masked load/store semantics,
tensor-core `nl.dot`, and the same "one program = one tile" mental model.

## How it works

Triton is a full MLIR/LLVM compiler. newt takes the shortest path that
preserves the programming model *and* the performance characteristics:

```
Python AST ──> typed block values (shape/dtype/layout)
           ──> CUDA C++ source           (compiler/codegen.py)
           ──> NVRTC, in-process JIT     (runtime/cuda.py, ctypes)
           ──> cubin ──> cuLaunchKernel  (no cuda-python, no nvcc subprocess)
```

The interesting part is mapping Triton's *block* semantics onto CUDA's
*thread* semantics:

| Triton concept | newt implementation |
|---|---|
| program instance | one thread block, `num_warps × 32` threads |
| block tensor | registers, **group-cyclic layout**: element *i* → thread `(i/VEC) % T`, so warps read coalesced *and* each thread owns 16-byte groups |
| `load`/`store` | runtime-checked vector fast path (`ld.global` 128-bit) with predicated scalar fallback - no static contiguity analysis needed |
| reductions | register partials → `__shfl_xor_sync` butterfly → smem across warps |
| broadcasting | numel-preserving = free; real broadcasts staged through a shared-memory arena |
| `nl.dot` | smem staging + **WMMA tensor cores** (fp16/bf16 → hmma, fp32 → tf32 like Triton's default); accumulator lives in fragments across k-loops, converted to/from register layout on demand (band-by-band to keep the smem footprint tiny) |
| `constexpr` | compile-time folding + dead-branch pruning |
| JIT cache | specialization on (constexprs, arg dtypes, num_warps), in-memory + on-disk cubin cache |

Everything lives in five files you can read in an afternoon:

```
newt/language.py           the nl.* DSL surface (mirrors triton.language)
newt/compiler/types.py     dtypes, pointers, promotion, broadcasting
newt/compiler/codegen.py   AST -> CUDA C++ (the compiler)
newt/runtime/cuda.py       ctypes NVRTC + CUDA driver bindings
newt/runtime/jit.py        @newt.jit, specialization, launch
newt/runtime/autotuner.py  @newt.autotune / @newt.heuristics
```

## What's supported

`program_id` `num_programs` `arange` `zeros` `full` `load` `store`
(masks + `other`), full arithmetic/comparison/bitwise ops with numpy-style
broadcasting, `where` `maximum` `minimum` `fma`, `exp` `log` `exp2` `log2`
`sqrt` `rsqrt` `sin` `cos` `tanh` `erf` `sigmoid` `abs` `floor` `ceil`,
`sum` `max` `min` (full + axis), `dot` (+accumulator), `.to()` casts,
`expand_dims` / `x[:, None]`, `reshape` `trans` `broadcast_to`,
`atomic_add` `atomic_max`, `cdiv` `static_assert` `static_print`,
`for range()` / `while` / `if` with constexpr pruning, tuple unpacking,
fp32 / fp16 / bf16 / fp64 / int8-64 / uint / bool, grids up to 3D,
`num_warps` 1-32, `@autotune` / `@heuristics`.

See `examples/` (vector add → fused softmax → layernorm → autotuned matmul →
**fused flash attention**) and 160+ tests in `tests/`.

## Performance

Memory-bound kernels match Triton (same coalesced, vectorized accesses);
matmul reaches a solid fraction of Triton on tensor cores - Triton's
remaining edge is its multi-stage `cp.async` software pipelining
(`num_stages`), which newt accepts-but-ignores. Run it yourself:

```
python benchmarks/bench.py            # newt vs triton-windows vs torch
```

Measured on an RTX PRO 5000 Blackwell Laptop GPU (sm_120), idle, identical
kernel source and config sweep for newt and triton-windows
(full tables: `benchmarks/results.md`):

| kernel | torch | **newt** | triton |
|---|---|---|---|
| vector add 64M (GB/s) | 784 | **780** | 774 |
| fused softmax 4096×8192 (GB/s) | 634 | **634** | 635 |
| layernorm 4096×8192 (GB/s) | 622 | **767** | 635 |
| matmul fp16 4096³ (TFLOP/s) | 101.9 | **66.2** | 107.5 |
| matmul tf32 8192³ (TFLOP/s) | 59.1 | **14.6** | 10.9 |

Memory-bound kernels sit at parity with Triton. Tensor-core matmul reaches
~50-70 % of Triton - the chunked `cp.async` pipeline hides most intra-tile
load latency, but Triton's cross-iteration multi-stage pipelining and
`mma.sync`+swizzled-smem codegen keep an edge that a mini can acknowledge
rather than chase.

## Known limitations (by design, it's a mini)

- Block dims must be powers of two (like `tl.arange`).
- `num_stages` pipelining is a no-op; `tl.rand`/philox, `device_print`,
  calling other `@jit` functions, and multi-dim `reshape` tricks are omitted.
- `/` `%` `//` on integer blocks follow C truncation semantics.
- Pointer offsets are int32 (tensors < 2³¹ elements).
- fp32 `dot` always uses tf32 tensor cores (Triton's default too).

## Install

```
pip install -e .           # needs torch + an NVIDIA GPU + CUDA toolkit (NVRTC)
python -m pytest tests -q
```

Works on Windows (developed on one - NVRTC DLL discovery included) and Linux.
