import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


@newt.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, stride, eps, BLOCK_N: nl.constexpr):
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


def test_softmax():
    for M, N in [(8, 128), (128, 1024), (32, 781)]:
        x = torch.randn(M, N, device="cuda")
        out = torch.empty_like(x)
        BLOCK_N = newt.next_power_of_2(N)
        softmax_kernel[(M,)](x, out, N, x.stride(0), BLOCK_N=BLOCK_N)
        torch.cuda.synchronize()
        ref = torch.softmax(x, dim=-1)
        newt.testing.assert_close(out, ref, msg=f"softmax {M}x{N}")
    print("softmax OK")


def test_layernorm():
    for M, N in [(8, 128), (64, 2048), (16, 300)]:
        x = torch.randn(M, N, device="cuda")
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out = torch.empty_like(x)
        BLOCK_N = newt.next_power_of_2(N)
        layernorm_kernel[(M,)](x, w, b, out, N, x.stride(0), 1e-5, BLOCK_N=BLOCK_N)
        torch.cuda.synchronize()
        ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps=1e-5)
        newt.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4, msg=f"layernorm {M}x{N}")
    print("layernorm OK")


if __name__ == "__main__":
    test_softmax()
    test_layernorm()
