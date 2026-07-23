"""Matmul on tensor cores with autotuning.

The classic Triton tutorial kernel: 2D grid of output tiles, pointer blocks
walked along K, fp32 accumulator held in WMMA fragments. @newt.autotune picks
the tile shape / warp count per (M, N, K).
"""

import torch

import newt
import newt.language as nl


@newt.autotune(
    configs=[
        newt.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        newt.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        newt.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        newt.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@newt.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: nl.constexpr, BLOCK_N: nl.constexpr, BLOCK_K: nl.constexpr):
    pid_m = nl.program_id(0)
    pid_n = nl.program_id(1)
    offs_m = pid_m * BLOCK_M + nl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + nl.arange(0, BLOCK_N)
    offs_k = nl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = nl.zeros((BLOCK_M, BLOCK_N), dtype=nl.float32)
    for k in range(nl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        a = nl.load(a_ptrs, mask=a_mask, other=0.0)
        b = nl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = nl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    nl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (newt.cdiv(M, meta["BLOCK_M"]), newt.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c


def main():
    M, N, K = 1024, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = matmul(a, b)
    ref = a @ b
    rel = ((c.float() - ref.float()).abs().max() /
           ref.float().abs().max()).item()
    print(f"matmul fp16 {M}x{N}x{K}: max rel err = {rel:.2e}")
    assert rel < 1e-2
    print("autotune picked:", matmul_kernel.best_config)

    tflops = 2 * M * N * K / 1e12
    t_newt = newt.testing.do_bench(lambda: matmul(a, b))
    t_torch = newt.testing.do_bench(lambda: a @ b)
    print(f"newt {t_newt:.3f} ms ({tflops/t_newt*1e3:.1f} TF) | "
          f"torch {t_torch:.3f} ms ({tflops/t_torch*1e3:.1f} TF) "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
