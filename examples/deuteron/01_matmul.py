"""deuteron matmul: write tiles, get an autotuned tensor-core kernel.

The function below is also runnable as plain PyTorch (matmul.ref) - that
eager mode is the oracle the autotuner validates candidate configs against.
"""

import torch

import deuteron as dt


@dt.kernel(verbose=True)
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc


def main():
    M, K, N = 2048, 1024, 2048
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    y = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)

    matmul(x, y, out)  # first call: traces, autotunes, caches
    ref = (x.float() @ y.float()).half()
    err = (out.float() - ref.float()).abs().max().item()
    print(f"\nmatmul {M}x{K}x{N}: max abs err = {err:.3e}")
    print("best config:", dict(matmul.best_config))
    print("\n--- generated newt kernel ---")
    print(matmul.to_newt_source(x, y, out))

    from newt.testing import do_bench

    t_dt = do_bench(lambda: matmul(x, y, out))
    t_torch = do_bench(lambda: torch.matmul(x, y, out=out))
    tf = 2 * M * N * K / 1e12
    print(f"deuteron {t_dt:.3f} ms ({tf/t_dt*1e3:.1f} TF) | "
          f"torch {t_torch:.3f} ms ({tf/t_torch*1e3:.1f} TF) "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
