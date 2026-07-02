# Overview

## The problem statement

Modern GPU kernels for ML are increasingly written in **high-level Python
DSLs** instead of raw CUDA:

- **[Triton](https://github.com/triton-lang/triton)** (OpenAI) lets you write
  a kernel as a Python function over *blocks* (tiles of data). One decorated
  function replaces hundreds of lines of CUDA: the compiler handles thread
  mapping, memory coalescing, shared memory, and tensor cores. It powers a
  large share of the custom kernels in today's LLM stacks.
- **[Helion](https://github.com/pytorch/helion)** (PyTorch) sits one level
  higher: you write PyTorch-like tile code with *no* kernel-level details at
  all, and it compiles down to Triton and **autotunes** the tile sizes and
  launch parameters automatically.

Both are large industrial compilers (Triton rides on MLIR/LLVM; Helion on
Triton + a search infrastructure). Their *ideas*, however, are compact - and
the best way to understand and own those ideas is to rebuild them small.
That is exactly what [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
did for vLLM: a ~1.2k-line reimplementation with comparable throughput that
people can actually read.

**Goal of this repo**: do the same for the Triton/Helion stack -

1. a mini-Triton (**newt**) with the full core programming model and
   performance comparable to real Triton on the kernels it supports, and
2. a mini-Helion (**deuteron**) that generates and autotunes newt kernels,
   with correctness guaranteed against an eager PyTorch oracle.

## How we solve it

### newt - mini-Triton (`newt/`)

Triton's pipeline is `Python AST → Triton IR → MLIR → LLVM → PTX`. Carrying
MLIR is impossible for a mini project, so newt takes the shortest path to
real machine code that *preserves the programming model and the performance
characteristics*:

```
@newt.jit Python fn ──ast──> typed block values (shape/dtype/layout)
                    ──emit──> CUDA C++ source
                    ──NVRTC──> cubin        (in-process JIT, ctypes)
                    ──driver──> cuLaunchKernel on torch's stream
```

nvcc/NVRTC does register allocation and instruction scheduling; newt's
compiler concentrates on the one hard, interesting problem - mapping
Triton's *block semantics* onto CUDA's *thread semantics*:

- **one program instance = one thread block** (`num_warps × 32` threads);
- block tensors live in **registers, group-cyclic across threads** - element
  *i* belongs to thread `(i/VEC) % T` - so warp accesses coalesce *and* each
  thread owns 16-byte groups that vectorize into 128-bit loads (a runtime
  contiguity/alignment check picks the fast path, no static analysis);
- **reductions** run register partials → warp `shfl_xor` butterflies → a
  shared-memory hop across warps;
- **broadcasting** between different-sized blocks stages through a single
  reusable shared-memory arena;
- **`nl.dot` uses tensor cores** (WMMA: fp16/bf16 → hmma, fp32 → tf32) with
  smem-staged tiles; when both operands come straight from `nl.load`, the
  tiles are streamed global→shared with **`cp.async` in K-chunks**, so the
  copy of chunk *k+1* overlaps the tensor-core math of chunk *k* - a mini
  version of Triton's `num_stages` pipelining;
- the accumulator lives in WMMA **fragments across loop iterations**, and
  converts to/from the register layout (band-by-band through smem) when the
  kernel does elementwise math on it - which is what makes a fused
  flash-attention forward possible (`examples/05_fused_attention.py`).

The API mirrors `triton.language` closely enough that porting a kernel is
usually `tl` → `nl`. Autotune/heuristics decorators, constexpr
specialization, masked loads/stores, atomics, grids up to 3D - all there.
~3k lines, 160 pytest tests, adversarially reviewed (see `LOG.md`).

### deuteron - mini-Helion (`deuteron/`)

Helion's central trick is that the *same function* is both the kernel spec
and its own reference implementation. deuteron replicates that:

```python
@dt.kernel
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):   # grid
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):                     # loop
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]       # tensor cores
        out[tile_m, tile_n] = acc
```

1. **Eager oracle** - run the function with full-size tiles: it's plain
   PyTorch (`matmul.ref(...)`), giving ground-truth outputs.
2. **Codegen** - trace the AST into a *newt kernel source string* (inspect
   with `matmul.to_newt_source(...)`; it looks exactly like the hand-written
   Triton tutorial kernel). Indexing tiles generates pointer math and
   boundary masks automatically, and masks propagate through expressions so
   reductions get correct identity fills (max → -inf, sum → 0) on padded
   lanes.
3. **Autotune** - every tile size is a constexpr; the tuner samples the
   config space, *rejects configs whose output disagrees with the oracle*,
   times survivors with CUDA events, pattern-searches around the best, and
   persists winners keyed by (source hash, shape bucket, dtypes).

## What "equivalent performance" looks like (RTX PRO 5000 Blackwell laptop)

Measured with `benchmarks/bench.py` - identical kernel source for newt
and triton-windows, same config sweep for both, idle GPU:

- **vector add / fused softmax / layernorm**: newt ≈ triton (softmax and
  layernorm at or above triton at every size measured) - memory-bound
  kernels are at parity, as they should be: both emit coalesced, vectorized
  accesses that saturate DRAM.
- **matmul fp16 (tensor cores)**: newt reaches ~50-70 % of triton after the
  cp.async pipeline (triton's remaining edge: deeper cross-iteration
  multi-stage pipelining, `mma.sync` with swizzled smem layouts - that
  machinery is Triton's moat and out of mini scope, documented in the
  README).
- **matmul tf32**: same ratio, and newt degrades more gracefully at 8192³.

See `benchmarks/results.md` for the current numbers.

## Repository map

```
newt/       mini-Triton package   (compiler: newt/compiler/codegen.py)
deuteron/   mini-Helion package   (codegen to newt: deuteron/codegen.py)
tests/      both frameworks, one pytest suite
examples/   newt examples + examples/deuteron/
benchmarks/ newt vs triton-windows vs torch
test.ipynb  NumPy-verified matmul walkthrough
PLAN.md     architecture decisions and milestones
LOG.md      chronological build log (what broke, what was learned)
```

## What's deliberately out of scope

Multi-stage cross-iteration pipelining (`num_stages`), `tl.rand`/Philox,
device-side printf, calling `@jit` functions from kernels, CUDA graphs,
non-NVIDIA backends, and Helion's full search space (loop reordering,
persistent kernels). Each is documented where it matters; none changes the
core ideas this repo exists to demonstrate.
