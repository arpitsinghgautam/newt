import torch

import deuteron as dt


@dt.kernel(config={"BLOCK_TILE_M": 64, "BLOCK_TILE_N": 64, "BLOCK_TILE_K": 32,
                   "num_warps": 4})
def matmul_fixed(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc


@dt.kernel
def matmul(x, y, out):
    for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):
        acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
        for tile_k in dt.tile(x.shape[1]):
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc


@dt.kernel
def softmax(x, out):
    for tile_m in dt.tile(x.shape[0]):
        row = x[tile_m, :]
        m = row.amax(1)
        e = dt.exp(row - m[:, None])
        s = e.sum(1)
        out[tile_m, :] = e / s[:, None]


@dt.kernel
def add_relu(x, y, out):
    for tile_n in dt.tile(x.shape[0]):
        out[tile_n] = dt.relu(x[tile_n] + y[tile_n])


def test_matmul_fixed_config():
    x = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    y = torch.randn(256, 384, device="cuda", dtype=torch.float16)
    out = torch.empty(512, 384, device="cuda", dtype=torch.float16)
    matmul_fixed(x, y, out)
    torch.cuda.synchronize()
    ref = (x.float() @ y.float()).half()
    assert torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print("deuteron matmul (fixed config) OK")
    print(matmul_fixed.to_newt_source(x, y, out))


def test_matmul_autotuned():
    x = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
    y = torch.randn(512, 768, device="cuda", dtype=torch.float16)
    out = torch.empty(1024, 768, device="cuda", dtype=torch.float16)
    matmul(x, y, out)
    torch.cuda.synchronize()
    ref = (x.float() @ y.float()).half()
    assert torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print("deuteron matmul (autotuned) OK, best:", dict(matmul.best_config))


def test_softmax():
    x = torch.randn(1000, 777, device="cuda")
    out = torch.empty_like(x)
    softmax(x, out)
    torch.cuda.synchronize()
    ref = torch.softmax(x, -1)
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-4)
    print("deuteron softmax OK, best:", dict(softmax.best_config))


def test_add_relu():
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")
    out = torch.empty_like(x)
    add_relu(x, y, out)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.relu(x + y))
    print("deuteron add_relu OK, best:", dict(add_relu.best_config))


def test_eager_reference():
    x = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    y = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    out = torch.empty(128, 32, device="cuda", dtype=torch.float16)
    matmul.ref(x, y, out)
    ref = x @ y
    assert torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print("deuteron eager reference OK")


if __name__ == "__main__":
    import os

    os.environ.setdefault("DEUTERON_VERBOSE", "1")
    test_eager_reference()
    test_matmul_fixed_config()
    test_softmax()
    test_add_relu()
    test_matmul_autotuned()
