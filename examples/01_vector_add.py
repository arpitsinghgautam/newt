"""Vector addition - the "hello world" of newt.

Each program instance (CUDA thread block) handles BLOCK elements. The mask
keeps the tail safe when n isn't a multiple of BLOCK. Compare with the Triton
tutorial: the kernel is identical modulo `tl` -> `nl`.
"""

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


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (newt.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=4096)
    return out


def main():
    n = 1 << 24
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = add(x, y)
    err = (out - (x + y)).abs().max().item()
    print(f"vector add n={n}: max err = {err:.2e}")
    assert err == 0.0

    t = newt.testing.do_bench(lambda: add(x, y))
    gbps = 3 * n * 4 / t / 1e6
    print(f"newt: {t:.3f} ms ({gbps:.0f} GB/s)  (GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
