import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import newt
import newt.language as nl


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
    for k in range(0, nl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        a = nl.load(a_ptrs, mask=a_mask, other=0.0)
        b = nl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = nl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = acc.to(c_ptr.dtype.element_ty)
    nl.store(c_ptrs, c, mask=c_mask)


def run_matmul(M, N, K, dtype, BM=64, BN=64, BK=32, num_warps=4):
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.5
    b = torch.randn(K, N, device="cuda", dtype=dtype) * 0.5
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    grid = (newt.cdiv(M, BM), newt.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return a, b, c


def test_matmul_fp16():
    for M, N, K in [(64, 64, 32), (256, 256, 128), (300, 200, 100)]:
        a, b, c = run_matmul(M, N, K, torch.float16)
        ref = (a.float() @ b.float()).half()
        newt.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2, msg=f"fp16 {M}x{N}x{K}")
    print("matmul fp16 OK")


def test_matmul_bf16():
    a, b, c = run_matmul(256, 128, 64, torch.bfloat16)
    ref = (a.float() @ b.float()).bfloat16()
    newt.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2, msg="bf16")
    print("matmul bf16 OK")


def test_matmul_tf32():
    a, b, c = run_matmul(128, 128, 128, torch.float32)
    ref = a @ b
    newt.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2, msg="tf32")
    print("matmul tf32 OK")


if __name__ == "__main__":
    test_matmul_fp16()
    test_matmul_bf16()
    test_matmul_tf32()
