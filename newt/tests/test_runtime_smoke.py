"""Smoke test: NVRTC compile + driver launch on torch tensors, no compiler."""

import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from newt.runtime.cuda import Kernel, compile_cuda

SRC = r"""
extern "C" __global__ void saxpy(float a, const float* x, const float* y,
                                 float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}
"""


def test_saxpy():
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    cubin = compile_cuda(SRC, "saxpy", arch)
    k = Kernel(cubin, "saxpy")

    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    stream = torch.cuda.current_stream().cuda_stream
    k.launch(
        grid=((n + 255) // 256,),
        block=(256,),
        args=[
            ctypes.c_float(2.5),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int32(n),
        ],
        stream=stream,
    )
    torch.cuda.synchronize()
    assert torch.allclose(out, 2.5 * x + y, atol=1e-5)
    print("saxpy smoke test OK, max err:", (out - (2.5 * x + y)).abs().max().item())


if __name__ == "__main__":
    test_saxpy()
