# deuteron ⚛

A **nano-[Helion](https://github.com/pytorch/helion)**: write PyTorch-like
tile code, get autotuned GPU kernels. Helion compiles a tile DSL down to
Triton and autotunes the result; deuteron does exactly that one level down -
it compiles to [newt](../README.md) (the nano-Triton) and autotunes block sizes
and warp counts automatically.

> *Why "deuteron"? A helion is a helium-3 nucleus; a deuteron is the lighter
> two-particle nucleus in the same family.*

```python
import torch
import deuteron as dt

@dt.kernel
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):   # launch grid
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):                     # k-loop
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]       # tensor cores
        out[tile_m, tile_n] = acc

matmul(x, y, out)   # first call: trace -> generate newt -> autotune -> cache
```

No `program_id`, no offsets, no masks, no block sizes: indexing a tensor with
tiles generates the pointer math and boundary masks; `@` becomes a
tensor-core `nl.dot` with a fused accumulator; every tile size becomes a
tunable `constexpr`.

## The Helion recipe, miniaturized

1. **Eager oracle.** The kernel function runs as *plain PyTorch* when tiles
   are full-size - `matmul.ref(x, y, out)` - which is both an interpreter
   (`DEUTERON_INTERPRET=1`) and the ground truth the autotuner checks every
   candidate config against. Wrong-result configs are rejected, not shipped.
2. **Codegen.** The AST is traced into a newt kernel (inspect it with
   `matmul.to_newt_source(x, y, out)` - it looks exactly like the
   hand-written Triton tutorial kernel).
3. **Autotune.** Random sample over the config space + local pattern search
   (halve/double each block size, step `num_warps`), timed with CUDA events.
   Winners persist in `~/.deuteron/configs.json`, keyed by kernel source,
   shape bucket (next-pow-2), and dtypes.

Mask bookkeeping is automatic *through expressions*: in

```python
@dt.kernel
def softmax(x, out):
    for tile_m in dt.tile(x.shape[0]):
        row = x[tile_m, :]                 # ':' = whole dim, masked to N
        m = row.amax(1)                    # padded lanes filled with -inf
        e = dt.exp(row - m[:, None])
        out[tile_m, :] = e / e.sum(1)[:, None]   # sum fills 0
```

the load mask propagates through `exp`/arithmetic so each reduction gets the
correct identity fill - padded lanes can't corrupt a max or a variance (see
`examples/03_layernorm.py`).

## What's supported

grid tiles (1-3D) + sequential inner tiles + `:` full-dim tiles, tensor
loads/stores with automatic masks, `@` / `acc += a @ b` (fused into the dot
accumulator), `+ - * / **` and comparisons, `dt.exp/log/sqrt/rsqrt/sigmoid/
tanh/erf/abs/maximum/minimum/where/relu`, `.sum/.amax/.amin(axis)`,
`dt.zeros/dt.full`, scalar params, fp16/bf16/fp32.

Not supported (yet, it's a nano): `if` inside kernels, `torch.empty` inside
the kernel (pass outputs as arguments), indexing with slices/ints, `mean()`
(use `.sum(axis) / x.shape[i]`), multiple grid loops per kernel.

## Files

```
deuteron/language.py   dt.* surface + the eager (PyTorch) implementations
deuteron/codegen.py    tile-DSL AST -> newt kernel source
deuteron/runtime.py    tracing, eager oracle, autotuner, config cache
```

## Install

Ships with newt in the same distribution, from the repo root:

```
pip install -e ..                        # installs newt + deuteron
python -m pytest ../tests -q
python ../examples/deuteron/01_matmul.py   # prints the generated newt kernel
```
