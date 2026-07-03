# newt 🦎

A **mini-Triton**: the [Triton](https://github.com/triton-lang/triton) GPU
programming model - block-level kernels written in Python, JIT-compiled to
real GPU machine code - reimplemented from scratch in ~3,000 lines of
readable Python, in the spirit of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

The repo also contains **[deuteron](deuteron/)** ⚛, a mini-
[Helion](https://github.com/pytorch/helion) built on top of newt: a
PyTorch-like tile DSL that generates newt kernels and autotunes them
automatically. Together they replicate the modern two-layer kernel-DSL
stack in miniature:

```mermaid
flowchart LR
    subgraph mini["this repo"]
        D["deuteron<br/>tile DSL + autotuner"] --> N["newt<br/>block DSL + JIT compiler"]
        N --> CU["CUDA C++ -> NVRTC -> cubin<br/>(ctypes, in-process)"]
    end
    subgraph prod["production stack it mirrors"]
        H["Helion"] --> T["Triton"]
        T --> M["MLIR -> LLVM -> PTX"]
    end
    mini -.->|same ideas, 100x smaller| prod
```

> *Why "newt"? **Triton** was the original genus name for newts
> (Laurenti, 1768). A newt is literally a small triton. A deuteron is a
> lighter nucleus than a helion.*

## Quick taste

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

If you know Triton, you know newt: replace `tl` with `nl` and it usually
just runs. Same `@jit`/grid launch protocol, `constexpr` specialization,
`@autotune`/`@heuristics` decorators, masked load/store semantics,
tensor-core `nl.dot`, and the same "one program = one tile" mental model.

And the same computation, one level up, in deuteron:

```python
import deuteron as dt

@dt.kernel
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):   # launch grid
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):                     # k-loop
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]       # tensor cores
        out[tile_m, tile_n] = acc

matmul(x, y, out)   # traces, generates a newt kernel, autotunes, caches
```

## What happens when you call a newt kernel

```mermaid
flowchart TD
    A["kernel[grid](args, BLOCK=..., num_warps=...)"] --> B["classify arguments<br/>tensors -> typed pointers<br/>ints/floats -> scalar params<br/>constexpr annotations -> compile-time values"]
    B --> C{"specialization cache hit?<br/>key = constexpr values + arg dtypes + num_warps"}
    C -- "hit" --> H["marshal ctypes arguments"]
    C -- "miss" --> D["walk the Python AST<br/>every value gets shape + dtype + layout<br/>(uniform scalar / register block / WMMA fragments)"]
    D --> E["emit CUDA C++<br/>group-cyclic register layout<br/>vectorized 16B load/store fast paths<br/>warp-shuffle reductions, smem broadcasting arena<br/>WMMA dot with chunked cp.async staging"]
    E --> F["NVRTC compiles to a cubin<br/>(in-process, disk-cached in ~/.newt/cache)"]
    F --> G["cuModuleLoadData + cuModuleGetFunction<br/>(CUDA driver API via ctypes)"]
    G --> H
    H --> I["cuLaunchKernel on torch's current stream<br/>grid blocks x (num_warps * 32) threads"]
```

The compiler's whole job is mapping Triton's *block* semantics onto CUDA's
*thread* semantics:

| Triton concept | newt implementation |
|---|---|
| program instance | one thread block, `num_warps × 32` threads |
| block tensor | registers, **group-cyclic layout**: element *i* lives in thread `(i/VEC) % T`, so warp accesses coalesce and each thread owns 16-byte groups |
| `load`/`store` | runtime-checked vector fast path (128-bit `ld.global`) with predicated scalar fallback; no static contiguity analysis needed |
| reductions | register partials -> `__shfl_xor_sync` butterfly -> smem across warps |
| broadcasting | numel-preserving reshapes are free; real broadcasts stage through a shared-memory arena |
| `nl.dot` | smem staging + **WMMA tensor cores** (fp16/bf16 -> hmma, fp32 -> tf32); operands coming straight from `nl.load` stream global->shared with **`cp.async` in K-chunks**, overlapping the copy of chunk k+1 with the math of chunk k |
| `constexpr` | compile-time folding + dead-branch pruning |
| JIT cache | specialization on (constexprs, arg dtypes, num_warps), in-memory + on-disk cubin cache |

## What happens when you call a deuteron kernel

```mermaid
flowchart TD
    A["matmul(x, y, out)"] --> B["trace the kernel AST<br/>dt.tile loops -> grid dims / k-loops<br/>tile indexing -> pointer math + boundary masks<br/>@ -> nl.dot with fused accumulator"]
    B --> C["generate newt kernel source<br/>(every tile size is a constexpr)<br/>exec -> @newt.jit function"]
    C --> D{"config cache hit?<br/>key = source hash + shape bucket + dtypes"}
    D -- "hit" --> K["launch best config through newt"]
    D -- "miss" --> E["eager oracle<br/>run the SAME function as plain PyTorch<br/>on cloned inputs -> ground truth"]
    E --> F["sample candidate configs<br/>block sizes x num_warps"]
    F --> G["run each candidate on clones"]
    G --> H{"output matches the oracle?"}
    H -- "no" --> R["reject config"] --> F
    H -- "yes" --> I["time with CUDA events"]
    I --> J["pattern search around the best<br/>halve/double each block, step num_warps"]
    J --> P["persist winner<br/>~/.deuteron/configs.json"]
    P --> K
```

The oracle step is the Helion trick that makes autotuning safe: a config
that compiles and runs but computes the wrong thing is rejected before it
is ever timed. `matmul.ref(x, y, out)` runs the oracle directly, and
`matmul.to_newt_source(x, y, out)` prints the generated kernel (it looks
exactly like the hand-written Triton tutorial matmul).

## Performance

Measured on an RTX PRO 5000 Blackwell Laptop GPU (sm_120), idle, identical
kernel source and config sweep for newt and triton-windows (full tables:
[benchmarks/results.md](benchmarks/results.md); rerun with
`python benchmarks/bench.py`):

| kernel | torch | **newt** | triton |
|---|---|---|---|
| fused softmax 4096×8192 (GB/s) | 760 | **765** | 767 |
| layernorm 4096×8192 (GB/s) | 625 | **767** | 764 |
| matmul fp16 2048³ (TFLOP/s) | 105.7 | **83.4** | 100.1 |
| matmul fp16 8192³ (TFLOP/s) | 107.1 | **80.1** | 103.7 |
| matmul tf32 8192³ (TFLOP/s) | 61.5 | **22.1** | 53.2 |

(Vector add sits at parity too: 475-785 GB/s depending on the thermal
state, always within a few percent of triton in the same run. Cold-start
single-kernel runs reach 96 TFLOP/s fp16 at 4096³; this laptop throttles
sustained suites, so within-run comparisons are what count.)

**Why the memory-bound kernels hit parity.** Add, softmax and layernorm are
DRAM-bandwidth-bound: the winner is whoever issues wide, coalesced memory
transactions and keeps enough threads in flight, and nothing else matters.
newt gets there by construction: the group-cyclic layout makes every warp
access coalesced, the runtime-checked fast path turns loads into the same
128-bit vector instructions Triton emits, fusion keeps each row in
registers for one read and one write, and NVRTC schedules the resulting
straight-line C++ as well as Triton's LLVM pipeline schedules its IR. Once
both compilers saturate the memory bus there is no headroom left to differ,
which is also why newt occasionally wins (a slightly better num_warps pick
at a given size).

**Why matmul fp16 is at ~75-85%.** Matmul is compute-bound: the winner is
whoever keeps the tensor cores fed *every* cycle. newt fights with the same
weapons Triton uses. fp16/bf16 dots compile to raw PTX: `ldmatrix` fragment
loads from XOR-swizzled (bank-conflict-free, unpadded) shared memory feeding
`mma.sync.m16n8k16`, with the documented accumulator register mapping used
for every layout conversion. Memory latency hides behind an N-slot
`cp.async` ring (`num_stages`, default 3, a real tuning knob here just like
in Triton): each `nl.dot` execution streams its tile in asynchronously and
runs the mma for a tile staged iterations earlier, one block barrier per
k-step, with a deferred flush consuming the final tiles at the
accumulator's first downstream read. This stack took newt from ~45-70% of
Triton (naive WMMA) to ~75-85% sustained. The last stretch is Triton's
finest-grained scheduling: register double-buffering of fragments inside
the k-step, per-iteration address strength-reduction, and warp
specialization. tf32 stays on the WMMA path (~45%): its small K-fragments
leave less math per tile to hide latency behind.

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

See [examples/](examples/) (vector add -> fused softmax -> layernorm ->
autotuned matmul -> **fused flash attention**), 160+ tests in
[tests/](tests/), and [test.ipynb](test.ipynb) for a NumPy-verified
walkthrough of both frameworks.

## Layout

```
newt/language.py           the nl.* DSL surface (mirrors triton.language)
newt/compiler/types.py     dtypes, pointers, promotion, broadcasting
newt/compiler/codegen.py   AST -> CUDA C++ (the compiler)
newt/runtime/cuda.py       ctypes NVRTC + CUDA driver bindings
newt/runtime/jit.py        @newt.jit, specialization, launch
newt/runtime/autotuner.py  @newt.autotune / @newt.heuristics
deuteron/                  the mini-Helion package (own README inside)
tests/                     both frameworks, one pytest suite
examples/                  newt examples + examples/deuteron/
benchmarks/                newt vs triton-windows vs torch
test.ipynb                 NumPy-verified matmul walkthrough
```

## Known limitations (by design, it's a mini)

- Block dims must be powers of two (like `tl.arange`).
- Cross-iteration `num_stages` pipelining is a no-op (accepted, ignored);
  `tl.rand`/philox, `device_print`, and calling other `@jit` functions are
  omitted.
- `/` `%` `//` on integer blocks follow C truncation semantics.
- Pointer offsets are int32 (tensors < 2³¹ elements).
- fp32 `dot` always uses tf32 tensor cores (Triton's default too).

## Install

```
pip install -e .             # installs newt + deuteron
                             # (needs torch + NVIDIA GPU + CUDA toolkit)
python -m pytest tests -q    # 165 tests, both frameworks
```

Works on Windows (developed on one; NVRTC DLL discovery included) and Linux.
See [OVERVIEW.md](OVERVIEW.md) for the problem statement and approach,
[PLAN.md](PLAN.md) for architecture decisions, and [LOG.md](LOG.md) for the
build log.
