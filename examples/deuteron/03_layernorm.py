"""deuteron layernorm: mask propagation makes the variance correct even for
padded (non-power-of-two) rows without any explicit mask handling."""

import torch

import deuteron as dt


@dt.kernel(verbose=True)
def layernorm(x, w, b, out, eps):
    for tile_m in dt.tile(x.shape[0]):
        row = x[tile_m, :]
        mean = row.sum(1) / x.shape[1]
        diff = row - mean[:, None]
        var = (diff * diff).sum(1) / x.shape[1]
        rstd = dt.rsqrt(var + eps)
        out[tile_m, :] = diff * rstd[:, None] * w[:] + b[:]


def main():
    M, N = 4096, 1500
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    layernorm(x, w, b, out, 1e-5)
    ref = torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5)
    err = (out - ref).abs().max().item()
    print(f"\nlayernorm {M}x{N}: max err = {err:.3e}")
    print("best config:", dict(layernorm.best_config))
    assert err < 1e-3


if __name__ == "__main__":
    main()
