"""deuteron.language - the tile DSL surface, mirroring helion's `hl`.

Every function has a real *eager* implementation operating on torch tensors:
running a @deuteron.kernel function with full-size tiles executes it as plain
PyTorch. That eager mode is the correctness oracle the autotuner checks
candidate configs against (exactly Helion's trick), and doubles as an
interpreter (DEUTERON_INTERPRET=1).
"""

import torch

float32 = torch.float32
float16 = torch.float16
bfloat16 = torch.bfloat16
int32 = torch.int32
int64 = torch.int64


class Tile:
    """A tile of an iteration dimension. In eager mode it covers the whole
    dimension; in compiled kernels it becomes BLOCK_<name> + offsets + mask."""

    def __init__(self, size, name=None):
        self.size = int(size)
        self.name = name

    def as_slice(self):
        return slice(0, self.size)

    def __repr__(self):
        return f"Tile({self.name or '?'}, {self.size})"


def _conv_index(idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    return tuple(i.as_slice() if isinstance(i, Tile) else i for i in idx)


class EagerTensor:
    """Wraps a tensor arg during eager (reference) execution so Tile indices
    behave like full slices. Slicing returns plain tensors; stores write
    through to the underlying tensor."""

    def __init__(self, t):
        self._t = t

    def __getitem__(self, idx):
        return self._t[_conv_index(idx)]

    def __setitem__(self, idx, value):
        self._t[_conv_index(idx)] = value

    @property
    def shape(self):
        return self._t.shape

    @property
    def dtype(self):
        return self._t.dtype

    def size(self, d=None):
        return self._t.size() if d is None else self._t.size(d)


def tile(sizes):
    """Iterate over tiles of one or more dimensions.

    The outermost `for ... in dt.tile(...)` in a kernel becomes the launch
    grid; inner ones become sequential loops over tiles. In eager mode this
    yields exactly one full-size tile (or tuple of tiles).
    """
    if isinstance(sizes, (list, tuple)):
        yield tuple(Tile(s) for s in sizes)
    else:
        yield Tile(sizes)


def _t(x):
    return x  # eager passthrough marker


def zeros(shape, dtype=torch.float32):
    sizes = [s.size if isinstance(s, Tile) else int(s) for s in shape]
    return torch.zeros(*sizes, dtype=dtype, device="cuda")


def full(shape, value, dtype=torch.float32):
    sizes = [s.size if isinstance(s, Tile) else int(s) for s in shape]
    return torch.full(sizes, value, dtype=dtype, device="cuda")


# elementwise - eager implementations are just torch; the compiler maps the
# *names* to nl.* equivalents
exp = torch.exp
log = torch.log
sqrt = torch.sqrt
rsqrt = torch.rsqrt
sigmoid = torch.sigmoid
tanh = torch.tanh
erf = torch.erf
abs = torch.abs
maximum = torch.maximum
minimum = torch.minimum
where = torch.where


def relu(x):
    return torch.maximum(x, torch.zeros((), dtype=x.dtype, device=x.device))


# name -> newt expression template (args substituted positionally)
NEWT_FNS = {
    "exp": "nl.exp({0})",
    "log": "nl.log({0})",
    "sqrt": "nl.sqrt({0})",
    "rsqrt": "nl.rsqrt({0})",
    "sigmoid": "nl.sigmoid({0})",
    "tanh": "nl.tanh({0})",
    "erf": "nl.erf({0})",
    "abs": "nl.abs({0})",
    "maximum": "nl.maximum({0}, {1})",
    "minimum": "nl.minimum({0}, {1})",
    "where": "nl.where({0}, {1}, {2})",
    "relu": "nl.maximum({0}, 0.0)",
}

# reduction method name -> (newt fn, identity fill for masked lanes)
REDUCTIONS = {
    "sum": ("nl.sum", "0.0"),
    "max": ("nl.max", "float('-inf')"),
    "min": ("nl.min", "float('inf')"),
    "amax": ("nl.max", "float('-inf')"),
    "amin": ("nl.min", "float('inf')"),
    "mean": None,  # handled specially (sum / size)
}

DTYPE_NAMES = {
    torch.float32: "nl.float32",
    torch.float16: "nl.float16",
    torch.bfloat16: "nl.bfloat16",
    torch.int32: "nl.int32",
    torch.int64: "nl.int64",
}
