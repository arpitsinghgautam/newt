# The whole story: newt and deuteron, explained from zero

This document explains the entire project to someone with basic computer
science knowledge and **no background in GPUs, CUDA, Triton, or machine
learning systems**. Every acronym is expanded the first time it appears.
If you prefer pictures, open [OVERVIEW.html](OVERVIEW.html) for the same
story with diagrams. If you already know what Triton is, skip to
[Part 2](#part-2-the-problem-statement).

---

## Part 1: A primer, from CPUs to Triton

### 1.1 Why GPUs exist

A CPU (central processing unit) has a handful of very smart cores. Each one
runs one instruction stream very fast, predicts branches, reorders
instructions, and keeps big caches nearby. That design is perfect for code
full of decisions: parsers, databases, operating systems.

A GPU (graphics processing unit) makes the opposite bet. Instead of 8-16
smart cores it has thousands of simple ones, grouped into a few dozen
SMs (streaming multiprocessors, the GPU's version of a core cluster). It
wins whenever you need to do the *same simple thing to millions of numbers
at once*: shading pixels, and, it turns out, multiplying the matrices
inside neural networks. The machine this project was built on has roughly
75 times more raw arithmetic throughput in its GPU than in its CPU.

The catch: to get that throughput you must phrase your problem as
thousands of tiny identical workers, and you must feed those workers data
fast enough. Almost everything in GPU programming is about the second
part.

### 1.2 What a "kernel" is

A kernel is simply a function that runs on the GPU, launched over many
workers at once. When you write

```
c[i] = a[i] + b[i]      for every i in a million elements
```

as a kernel, you do not write the loop. You write the body once, and you
launch a *grid* of workers where worker number `i` handles element `i`
(or, more commonly, a *block* of elements). The GPU hardware runs
thousands of these workers concurrently.

Two more terms you will see constantly:

- A **thread** is one worker. Threads are grouped into **warps** (groups
  of 32 threads that execute in lockstep, i.e. they all run the same
  instruction at the same moment) and warps are grouped into **thread
  blocks** (up to ~1024 threads that can cooperate and share fast
  memory). One thread block runs on one SM.
- The **grid** is the collection of all thread blocks in one launch.

### 1.3 The memory pyramid (the actual game)

GPU arithmetic is nearly free; moving data is what costs time. The memory
system is a pyramid:

| level | size | speed | who sees it |
|---|---|---|---|
| registers | ~256 KB per SM | instant | one thread |
| shared memory ("smem") | ~100 KB per SM | a few cycles | one thread block |
| L2 cache | tens of MB | tens of cycles | whole GPU |
| DRAM / HBM (the GPU's main memory) | 8-80 GB | ~400-600 cycles | whole GPU |

Two consequences drive every design decision in this project:

1. **Coalescing.** When the 32 threads of a warp read 32 *adjacent*
   numbers, the hardware fetches them in one wide transaction. If they
   read 32 scattered numbers, it needs up to 32 transactions. Arranging
   data so warps always touch adjacent memory is called coalescing, and
   it is the difference between using 100% and 10% of your memory
   bandwidth.
2. **Latency hiding.** A DRAM read takes ~500 cycles. The GPU copes by
   having far more threads resident than executing: while one warp waits
   for memory, another computes. When a kernel cannot keep enough work in
   flight, the compute units sit idle waiting for data.

### 1.4 Why machine learning needs custom kernels

A neural network layer might compute `softmax(x)` (normalize a row of
numbers so they sum to 1). Done naively with library calls, that is three
separate passes over the data: find the max, subtract-and-exponentiate,
divide by the sum. Each pass reads the whole array from DRAM and writes it
back: six trips through the slowest memory.

A *fused* kernel does all three steps in one pass: load the row once into
registers, do all the math on-chip, write once. Same arithmetic, one third
of the memory traffic, roughly three times faster for this
memory-dominated operation. Modern ML performance work is largely the art
of fusing operations into single kernels, and that requires *writing*
kernels rather than composing library calls.

### 1.5 CUDA, and why writing it is hard

CUDA (NVIDIA's GPU programming platform) lets you write kernels in a C++
dialect. It is powerful and entirely manual: you pick each thread's
indices, manage shared memory by hand, place synchronization barriers,
arrange coalescing yourself, and invoke tensor cores (dedicated
matrix-multiply hardware inside each SM, ~16x faster than ordinary
arithmetic for this job) through special intrinsics. A production-quality
CUDA matrix multiply is hundreds of lines, and a single misplaced barrier
gives silently wrong answers or a deadlock.

### 1.6 Triton: kernels at the block level

[Triton](https://github.com/triton-lang/triton) (an open-source compiler,
originally from OpenAI) changed who can write fast kernels. In Triton you
program at the level of a **block of data**, not a thread. Here is a real,
complete Triton-style softmax kernel:

```python
@jit
def softmax_kernel(x_ptr, out_ptr, N, stride, BLOCK_N: constexpr):
    row = program_id(0)                    # which row this instance handles
    cols = arange(0, BLOCK_N)              # a whole block of column indices
    mask = cols < N                        # guard the ragged edge
    x = load(x_ptr + row * stride + cols, mask=mask, other=-inf)
    x = x - max(x, axis=0)                 # the whole row, in registers
    num = exp(x)
    nl_out = num / sum(num, axis=0)
    store(out_ptr + row * stride + cols, nl_out, mask=mask)
```

No threads, no warps, no shared memory, no barriers. You say "load this
block, reduce it, store it", and the *compiler* decides how the block's
elements are spread across threads, how the reduction uses warp
communication, when barriers are needed, and how to hit the tensor cores.
Triton kernels routinely match hand-written CUDA, and most of the custom
kernels in today's LLM (large language model) serving stacks are written
in it.

Under the hood Triton is a serious industrial compiler: Python is
translated into Triton's IR (intermediate representation, a
compiler-internal program format), then through MLIR (a compiler
framework) and LLVM (another one) down to PTX (NVIDIA's portable GPU
assembly), which NVIDIA's tools turn into SASS (the GPU's real machine
code). Hundreds of thousands of lines of code.

### 1.7 Helion: one level higher still

Even Triton makes you think about `program_id`, offsets, masks and block
sizes, and forces you to *tune* those block sizes per GPU.
[Helion](https://github.com/pytorch/helion) (from PyTorch) removes that
too: you write tile-level code that looks like PyTorch, and the compiler
generates the Triton kernel and **autotunes** it, automatically searching
over block sizes and other knobs, checking each candidate for correctness,
and caching the fastest configuration.

### 1.8 One more idea: JIT compilation

Everything above compiles kernels JIT (just-in-time): the kernel is
compiled the first time you call it with a particular combination of data
types and compile-time constants, then cached. This is what lets a Python
function become GPU machine code without a separate build step.

---

## Part 2: The problem statement

Triton and Helion are wonderful and enormous. You cannot realistically
read them. The best way to *own* the ideas inside a large system is to
rebuild it small: that is what
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) did for the vLLM
inference engine (~1,200 readable lines, comparable throughput), and it is
the founding idea here.

**Goal: rebuild the entire two-layer kernel-DSL stack, in miniature, with
real performance.** Concretely:

1. A mini-Triton, called **newt** (Triton was the original genus name for
   newts, so a newt is literally a small triton): the full block
   programming model, compiled to real GPU machine code, with performance
   comparable to actual Triton. Not a simulator, not a toy that only
   handles vector-add.
2. A mini-Helion, called **deuteron** (a deuteron is a lighter atomic
   nucleus than a helion): PyTorch-like tile code in, autotuned newt
   kernels out, with correctness guaranteed by construction.

Success criteria: memory-bound kernels at parity with real Triton
(they should be, if the compiler is doing its job), matrix multiplication
at a respectable fraction of Triton on tensor cores with an honest
account of the remaining gap, and a codebase a person can actually read
(~4,000 lines total).

---

## Part 3: How newt works

### 3.1 The shortest path to machine code

Triton's pipeline needs MLIR and LLVM. A mini cannot carry those, but it
does not need to, because NVIDIA ships a C++ compiler *as a library*:
NVRTC (NVIDIA Runtime Compilation). So newt's pipeline is:

```
your Python function
    -> Python AST (abstract syntax tree: the parsed structure of the code)
    -> typed "block values" (every value gets a shape, dtype and layout)
    -> a CUDA C++ source string        (newt/compiler/codegen.py)
    -> NVRTC compiles it in-process    (no files, no external compiler)
    -> a cubin (compiled GPU binary)
    -> loaded and launched via the CUDA driver API through ctypes
       (Python's built-in foreign-function interface: newt calls
       nvcuda.dll / libcuda.so directly, no wrapper packages)
```

The clever part of this choice: NVRTC does register allocation and
instruction scheduling, which are the person-decades parts of a compiler.
newt's ~2,500-line code generator only has to solve the one problem that
makes Triton *Triton*: translating block semantics into thread semantics
well. Everything below is about how it does that.

### 3.2 The layout: where a block actually lives

When a newt kernel says `x = nl.load(ptr + offs)` for a block of 1,024
floats, where are those floats? Answer: in **registers, spread across the
threads of the block** by one fixed rule, the *group-cyclic layout*:

```
element i  lives in  thread (i / VEC) mod T,  slot (i / (T*VEC)) * VEC + i mod VEC
```

where T is the thread count and VEC is a small group size (4 floats or
8 halves = 16 bytes). In plain words: consecutive elements are dealt out
to threads in 16-byte groups, round-robin.

Worked example with 8 elements, 2 threads, VEC=2:

```
element:  e0 e1 e2 e3 e4 e5 e6 e7
thread:   t0 t0 t1 t1 t0 t0 t1 t1     <- 2-element groups, round robin
```

This one rule buys three things simultaneously:

- **Coalescing by construction**: neighboring threads always hold
  neighboring memory, so every warp access is one wide transaction.
- **Vectorization**: each thread owns 16 contiguous bytes, so a load can
  be a single 128-bit instruction. newt does not *prove* contiguity at
  compile time the way Triton does; it emits a cheap runtime check per
  group ("are these offsets consecutive, aligned, all unmasked?") and
  branches to the fast path. A few integer compares against a ~500-cycle
  memory access is an excellent trade, and it made vector-add go from 82%
  of Triton to parity.
- **Simplicity**: elementwise math never cares where elements live; each
  thread just loops over its own slots.

### 3.3 Reductions and broadcasting

`nl.sum(x)` must combine values held by *different threads*. newt lowers
it in three stages: each thread reduces its own slots in registers; then
each warp reduces its 32 lanes with `__shfl_xor_sync` (a butterfly
exchange where lanes trade values directly, no memory involved, 5 steps
for 32 lanes); then one value per warp goes through shared memory and a
final warp finishes. This is exactly the shape of Triton's reduction.

Broadcasting (`a[:, None] + b[None, :]`) sometimes requires an element in
one thread to be visible to another. newt stages the smaller operand
through a reusable shared-memory scratch area (an "arena") with barriers
before and after. Cheap reshapes that do not move data (adding size-1
dimensions) are free, they only change metadata.

### 3.4 The matmul story (where the real performance lives)

Matrix multiplication is special for two reasons: tensor cores, and the
fact that it is compute-bound (the arithmetic outweighs the memory
traffic, so the game flips from "save bandwidth" to "never let the
multiply units stall"). newt's `nl.dot` went through three generations,
all still visible in the git history, and the final design has four
pieces:

**Tiles.** Each thread block computes one BM x BN tile of the output,
looping over K in BK-sized steps: load a BM x BK piece of A and a BK x BN
piece of B, multiply-accumulate, repeat. All three tile sizes are
`constexpr` (compile-time constants), which is what makes them tunable.

**Tensor cores via raw PTX.** For fp16/bf16 (16-bit floating point
formats), newt emits `ldmatrix` (an instruction where a warp
cooperatively loads 8x8 matrix fragments from shared memory into
registers) and `mma.sync.m16n8k16` (the tensor-core instruction: one warp
multiplies a 16x16 by a 16x8 fragment in one go) as inline PTX assembly.
The accumulator lives in registers with a documented per-lane mapping, so
newt can convert it to and from the normal layout whenever the kernel
does elementwise math on it (that conversion is what makes a fused
flash-attention kernel possible).

**Swizzled shared memory.** Shared memory is divided into 32 banks; if
several threads hit the same bank in one cycle they serialize (a "bank
conflict"). Naive tile layouts conflict badly under `ldmatrix`. newt
stores tiles *swizzled*: each row's 16-byte chunks are permuted by XORing
the chunk index with the row index. The permutation costs nothing (it is
just index arithmetic on both the writer and the reader side) and makes
every access pattern conflict-free. This replaces the padding trick used
by simpler kernels and shrinks the tiles at the same time.

**The pipeline ring.** The most important trick of all. A naive loop
stalls every iteration: load tile, wait ~500 cycles, compute, repeat.
Modern GPUs have `cp.async` (an instruction that copies global memory to
shared memory *in the background*, without the data passing through
registers). newt exploits it with an N-slot ring buffer and a deliberate
one-iteration delay:

```
iteration k:  if S-1 tiles are in flight:
                  wait for the OLDEST tile (staged S-1 iterations ago)
                  one barrier
                  run the tensor-core math for it
              start the background copy of THIS iteration's tile
```

While the tensor cores chew on an old tile, the copies for the next S-1
tiles are in flight. The number of slots S is the `num_stages` knob, the
same knob real Triton exposes, and the autotuner searches over it. A
subtle design point: the compiler cannot prefetch *future* tiles (their
addresses are computed later in the user's loop), so newt inverts the
problem and *delays consumption* instead, which achieves the same overlap
with no loop rewriting. The final staged tiles are consumed by a
"flush" emitted wherever the accumulator is next read.

**Fragment ping-pong.** One more level down: within one k-step, the
`ldmatrix` loads and the `mma` math also form a dependency chain. newt
double-buffers the fragments so that step k+1's loads are issued before
step k's math, hiding the shared-memory latency behind the tensor cores.
This last change alone took the cold-start matmul from 96 to 110
TFLOP/s (trillion floating-point operations per second).

### 3.5 The JIT layer

`@newt.jit` parses the function once. On every call it classifies the
arguments (tensors become typed pointers, Python numbers become scalar
parameters, `constexpr` annotations become compile-time values), forms a
specialization key (constants + dtypes + `num_warps` + `num_stages`), and
compiles on a miss. Compiled binaries are cached in memory and on disk,
so the second process to use a kernel pays nothing. Generated CUDA source
is one environment variable away (`NEWT_DEBUG=1`), which turned out to be
the single most useful debugging feature in the whole project.

---

## Part 4: How deuteron works

deuteron is ~700 lines that do to newt what Helion does to Triton. You
write:

```python
@dt.kernel
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc
```

No program ids, no offsets, no masks, no block sizes. Three mechanisms
make it work:

**Tracing.** deuteron walks the function's AST. The outer `dt.tile` loop
becomes the launch grid; inner ones become in-kernel loops; tensor
indexing like `x[tile_m, tile_k]` becomes pointer arithmetic plus
boundary masks; `@` becomes `nl.dot` fused into the accumulator. The
output is a *newt kernel as a source string* in which every tile size is
a `constexpr`. You can print it (`matmul.to_newt_source(...)`); it looks
exactly like the hand-written tutorial kernel, which is the most
satisfying demo in the repo.

**The eager oracle.** Here is Helion's key trick, replicated: the same
function also runs as *plain PyTorch* (tiles become full-size slices).
That gives ground-truth outputs for free. During autotuning, every
candidate configuration runs on cloned inputs and its result is compared
against the oracle; a config that compiles and runs but computes garbage
is rejected before it is ever timed. Autotuning can never ship a wrong
kernel.

**The search.** Candidates are sampled from the config space (block
sizes x `num_warps` x `num_stages`), correctness-filtered, timed with
CUDA events, refined by a local pattern search (halve/double each block
size, step the warp count), and the winner is persisted to disk keyed by
kernel, shape bucket and dtypes. Next call with similar shapes launches
instantly.

A small but pleasing detail: masks propagate through traced expressions,
so reductions automatically use the right identity value on padded lanes
(max fills with -inf, sum with 0). The layernorm example computes a
correct variance on rows of length 1,500 (not a power of two) with no
explicit mask handling anywhere in user code.

---

## Part 5: The results, and how to read them

Measured against triton-windows (real Triton) and torch (which calls
NVIDIA's hand-tuned cuBLAS/cuDNN libraries) on the same machine, same
kernel source, same tuning sweep:

| kernel | torch | newt | triton |
|---|---|---|---|
| fused softmax 4096x8192 (GB/s) | 760 | 765 | 767 |
| layernorm 4096x8192 (GB/s) | 625 | 767 | 764 |
| vector add 64M (GB/s) | 782 | 777 | 779 |
| matmul fp16 2048^3 (TFLOP/s) | 105.7 | 83.4 | 100.1 |
| matmul fp16 4096^3, cold (TFLOP/s) | ~100 | 109.7 | ~120 |
| matmul tf32 8192^3 (TFLOP/s) | 61.5 | 22.1 | 53.2 |

Why the two kinds of result?

- **Memory-bound kernels are at parity because there is nothing left to
  win.** Softmax reads and writes each byte once; the winner is whoever
  saturates DRAM bandwidth, and coalesced + vectorized + fused gets you
  there. Any correct compiler that manages those three ties. This is the
  "roofline" idea: performance is capped by min(compute peak, bandwidth x
  arithmetic intensity), and these kernels live on the bandwidth roof.
- **Matmul lives on the compute roof**, where every scheduling
  imperfection shows. newt reaches 76-83% of Triton sustained (~92% when
  both run cold; the test machine is a 110 W laptop that throttles under
  sustained load, so within-run comparisons are the honest ones). The
  remaining gap is Triton's finest-grained machinery: strength-reducing
  address computations across loop iterations and specializing warps into
  producer/consumer roles. Both are known, documented, and out of scope
  for a mini on purpose.
- **tf32 (a 19-bit float format used for fp32 matmuls on tensor cores)
  still uses the older WMMA path** (NVIDIA's higher-level tensor-core
  API, which hides the register layout and costs extra shared-memory
  round trips), so it sits near 45%. Porting it to the raw PTX path is
  mechanical future work.

Correctness got as much attention as speed: 176 tests compare every
operation against PyTorch; a notebook verifies both frameworks against
float64 NumPy; and three adversarial review campaigns (attacking the
compiler with hundreds of targeted GPU micro-programs, plus a symbolic
simulation of the pipeline state machine) each ended with every confirmed
finding fixed and regression-tested. The git history doubles as the build
log: each compiler stage is a self-contained commit.

---

## Part 6: What was deliberately left out

Random number generation inside kernels, device-side printing, calling
one `@jit` function from another, non-NVIDIA backends, fp8 formats,
Helion's larger search space (loop reordering, persistent kernels), and
the last few percent of matmul scheduling described above. Each omission
is documented where a user would hit it. None of them changes the ideas
this project exists to demonstrate: that the modern GPU kernel stack,
from tile-level Python down to tensor-core machine code, fits in four
thousand readable lines once you know which problems are essential and
which are incidental.

---

## Glossary (quick reference)

| term | meaning |
|---|---|
| kernel | a function that runs on the GPU across many parallel workers |
| thread / warp / block / grid | one worker / 32 lockstep workers / cooperating group with shared memory / all blocks in a launch |
| SM | streaming multiprocessor; the GPU's core cluster, runs whole thread blocks |
| DRAM / HBM | the GPU's large, slow main memory |
| shared memory (smem) | small fast per-block scratchpad, also the unit with banks |
| coalescing | adjacent threads touching adjacent memory so reads combine into wide transactions |
| bank conflict | multiple threads hitting the same shared-memory bank, serializing access |
| swizzle | XOR-based index permutation that removes bank conflicts without padding |
| tensor core | dedicated matrix-multiply hardware; driven by mma instructions |
| WMMA / mma.sync / ldmatrix | NVIDIA's high-level API / the raw tensor-core PTX instruction / the cooperative fragment-load instruction |
| cp.async | background copy from global to shared memory, bypassing registers |
| PTX / SASS / cubin | NVIDIA's portable GPU assembly / the real machine code / the compiled binary container |
| NVRTC | NVIDIA's C++-to-cubin compiler, usable as an in-process library |
| AST | abstract syntax tree; the parsed structure of source code |
| JIT | just-in-time compilation: compile on first use, cache afterwards |
| constexpr | a kernel parameter fixed at compile time; each value produces a specialized binary |
| DSL | domain-specific language, like the `nl.*` / `dt.*` mini-languages here |
| autotuning | automatically searching configuration knobs, keeping the fastest correct one |
| memory-bound / compute-bound | limited by bandwidth / limited by arithmetic throughput (the roofline model) |
