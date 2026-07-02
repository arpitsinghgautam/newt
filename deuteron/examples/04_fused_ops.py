"""Fused elementwise chains: one kernel instead of four torch launches."""

import torch

import deuteron as dt


@dt.kernel
def fused_gelu_residual(x, y, out):
    # out = gelu(x) + y, tanh approximation
    for t in dt.tile(x.shape[0]):
        v = x[t]
        inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        g = 0.5 * v * (1.0 + dt.tanh(inner))
        out[t] = g + y[t]


def main():
    n = 1 << 22
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    fused_gelu_residual(x, y, out)
    ref = torch.nn.functional.gelu(x, approximate="tanh") + y
    err = (out - ref).abs().max().item()
    print(f"fused gelu+residual n={n}: max err = {err:.3e}")
    print("best config:", dict(fused_gelu_residual.best_config))
    assert err < 1e-5

    from newt.testing import do_bench

    t_dt = do_bench(lambda: fused_gelu_residual(x, y, out))
    t_torch = do_bench(lambda: torch.nn.functional.gelu(x, approximate="tanh") + y)
    print(f"deuteron {t_dt:.3f} ms | torch eager {t_torch:.3f} ms "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
