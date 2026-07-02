import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import newt
import newt.language as nl


@newt.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: nl.constexpr):
    pid = nl.program_id(0)
    offs = pid * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask)
    nl.store(out_ptr + offs, x + y, mask=mask)


def test_add():
    for n in (1024, 1 << 20, 12345):  # incl. non-divisible size
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        grid = lambda meta: (newt.cdiv(n, meta["BLOCK"]),)
        add_kernel[grid](x, y, out, n, BLOCK=1024)
        torch.cuda.synchronize()
        assert torch.equal(out, x + y), f"mismatch for n={n}"
    print("vector_add OK")


if __name__ == "__main__":
    os.environ.setdefault("NEWT_DEBUG", "1")
    test_add()
