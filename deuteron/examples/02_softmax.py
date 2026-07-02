"""deuteron softmax: masks and identity-fills are inferred automatically.

`row.amax(1)` on padded lanes would be wrong with a plain 0-fill; deuteron
tracks the load mask through the expression and fills reductions with the
right identity (-inf for max, 0 for sum).
"""

import torch

import deuteron as dt


@dt.kernel(verbose=True)
def softmax(x, out):
    for tile_m in dt.tile(x.shape[0]):
        row = x[tile_m, :]
        m = row.amax(1)
        e = dt.exp(row - m[:, None])
        s = e.sum(1)
        out[tile_m, :] = e / s[:, None]


def main():
    x = torch.randn(8192, 3000, device="cuda")  # non-pow2 columns
    out = torch.empty_like(x)
    softmax(x, out)
    ref = torch.softmax(x, -1)
    err = (out - ref).abs().max().item()
    print(f"\nsoftmax {tuple(x.shape)}: max err = {err:.3e}")
    print("best config:", dict(softmax.best_config))

    from newt.testing import do_bench

    t_dt = do_bench(lambda: softmax(x, out))
    t_torch = do_bench(lambda: torch.softmax(x, -1))
    print(f"deuteron {t_dt:.3f} ms | torch {t_torch:.3f} ms "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
