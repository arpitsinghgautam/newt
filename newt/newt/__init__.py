"""newt - a mini-Triton: block-programming DSL JIT-compiled to CUDA.

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

    add_kernel[(newt.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
"""

from . import language
from . import testing
from .compiler.codegen import CompileError
from .runtime.autotuner import Autotuner, Config, Heuristics, autotune, heuristics
from .runtime.jit import JITFunction, jit

__version__ = "0.1.0"


def cdiv(a, b):
    return -(-a // b)


def next_power_of_2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


__all__ = [
    "jit", "autotune", "heuristics", "Config", "JITFunction", "Autotuner",
    "Heuristics", "CompileError", "language", "testing", "cdiv",
    "next_power_of_2", "__version__",
]
