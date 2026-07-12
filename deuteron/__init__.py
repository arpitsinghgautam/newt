"""deuteron - a nano-Helion: PyTorch-like tile DSL compiled to newt kernels
with automatic autotuning.

    import deuteron as dt

    @dt.kernel
    def matmul(x, y, out):
        for tile_m, tile_n in dt.tile([x.shape[0], y.shape[1]]):
            acc = dt.zeros([tile_m, tile_n], dtype=dt.float32)
            for tile_k in dt.tile(x.shape[1]):
                acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
            out[tile_m, tile_n] = acc

    matmul(x, y, out)          # traces, autotunes, caches, launches
    matmul.ref(x, y, out)      # eager PyTorch reference (the oracle)
    print(matmul.to_newt_source(x, y, out))  # inspect generated newt code
"""

from .language import (  # noqa: F401
    EagerTensor,
    Tile,
    abs,
    bfloat16,
    erf,
    exp,
    float16,
    float32,
    full,
    int32,
    int64,
    log,
    maximum,
    minimum,
    relu,
    rsqrt,
    sigmoid,
    sqrt,
    tanh,
    tile,
    where,
    zeros,
)
from .codegen import TraceError  # noqa: F401
from .runtime import Config, Kernel, kernel  # noqa: F401

__version__ = "0.1.0"
