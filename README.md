# optimal_kernels

Two sibling mini-frameworks replicating the modern GPU-kernel DSL stack,
nano-vllm style - small, readable, real performance:

| project | replicates | one-liner |
|---|---|---|
| **[newt](newt/)** 🦎 | [Triton](https://github.com/triton-lang/triton) | block-programming DSL → CUDA C++ → NVRTC → cubin, via ctypes; tensor-core `dot`, autotune, ~3k lines |
| **[deuteron](deuteron/)** ⚛ | [Helion](https://github.com/pytorch/helion) | PyTorch-like tile DSL → generates newt kernels → autotunes block sizes/warps with an eager-reference oracle |

The stack composes the same way the real one does:

```
deuteron (tiles, autotuning)  ~  Helion
   └─> newt (blocks, @jit)    ~  Triton
        └─> CUDA C++ / NVRTC / driver API
```

Start with [`newt/README.md`](newt/README.md), then
[`deuteron/README.md`](deuteron/README.md).
`PLAN.md` documents the architecture decisions; `LOG.md` is the build log.

Naming: *Triton* was the original genus name for newts - a newt is a small
triton. A *deuteron* is a lighter nucleus than a *helion*.
