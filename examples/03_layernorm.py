"""LayerNorm forward, one program per row.

Shows numerically-careful reductions (mean/variance via masked where) and
mixed elementwise math, matching torch.nn.functional.layer_norm.
"""

import torch

import newt
import newt.language as nl


@newt.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, stride, eps,
                     BLOCK_N: nl.constexpr):
    row = nl.program_id(0)
    cols = nl.arange(0, BLOCK_N)
    mask = cols < N
    x = nl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    mean = nl.sum(x, axis=0) / N
    diff = nl.where(mask, x - mean, 0.0)
    var = nl.sum(diff * diff, axis=0) / N
    rstd = nl.rsqrt(var + eps)
    w = nl.load(w_ptr + cols, mask=mask)
    b = nl.load(b_ptr + cols, mask=mask)
    y = (x - mean) * rstd * w + b
    nl.store(out_ptr + row * stride + cols, y, mask=mask)


def layernorm(x, w, b, eps=1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = newt.next_power_of_2(N)
    layernorm_kernel[(M,)](x, w, b, out, N, x.stride(0), eps,
                           BLOCK_N=BLOCK_N, num_warps=8)
    return out


def main():
    M, N = 4096, 3000
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    out = layernorm(x, w, b)
    ref = torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5)
    err = (out - ref).abs().max().item()
    print(f"layernorm {M}x{N}: max err = {err:.2e}")
    assert err < 1e-4

    t_newt = newt.testing.do_bench(lambda: layernorm(x, w, b))
    t_torch = newt.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5))
    print(f"newt {t_newt:.3f} ms | torch {t_torch:.3f} ms "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
