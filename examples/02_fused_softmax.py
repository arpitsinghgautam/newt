"""Fused row softmax.

One program per row; the whole row lives in registers, the max/sum reductions
run through warp shuffles + shared memory, and the row is normalized without
ever leaving the chip - three global-memory round trips (torch eager) fused
into one.
"""

import torch

import newt
import newt.language as nl


@newt.jit
def softmax_kernel(x_ptr, out_ptr, N, stride, BLOCK_N: nl.constexpr):
    row = nl.program_id(0)
    cols = nl.arange(0, BLOCK_N)
    mask = cols < N
    x = nl.load(x_ptr + row * stride + cols, mask=mask, other=float("-inf"))
    x = x - nl.max(x, axis=0)
    num = nl.exp(x)
    den = nl.sum(num, axis=0)
    nl.store(out_ptr + row * stride + cols, num / den, mask=mask)


def softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = newt.next_power_of_2(N)
    num_warps = 4 if BLOCK_N < 4096 else 8
    softmax_kernel[(M,)](x, out, N, x.stride(0), BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out


def main():
    x = torch.randn(4096, 2500, device="cuda")  # non-pow2 row length
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)
    err = (out - ref).abs().max().item()
    print(f"fused softmax {tuple(x.shape)}: max err = {err:.2e}")
    assert err < 1e-6

    t_newt = newt.testing.do_bench(lambda: softmax(x))
    t_torch = newt.testing.do_bench(lambda: torch.softmax(x, -1))
    print(f"newt {t_newt:.3f} ms | torch {t_torch:.3f} ms "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
