"""newt.language - the kernel DSL, mirroring triton.language.

Import as `import newt.language as nl`. These symbols are only meaningful
inside a @newt.jit function, where the compiler intercepts them at the AST
level; calling them from normal Python raises.
"""

from .compiler.types import (  # noqa: F401  (re-exported dtypes)
    bfloat16,
    float16,
    float32,
    float64,
    int1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint32,
)


class constexpr:
    """Annotation for compile-time constant kernel parameters."""

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        return cls

    def _unwrap(self, other):
        return other.value if isinstance(other, constexpr) else other

    def __add__(self, o):
        return constexpr(self.value + self._unwrap(o))

    def __radd__(self, o):
        return constexpr(self._unwrap(o) + self.value)

    def __sub__(self, o):
        return constexpr(self.value - self._unwrap(o))

    def __mul__(self, o):
        return constexpr(self.value * self._unwrap(o))

    def __rmul__(self, o):
        return constexpr(self._unwrap(o) * self.value)

    def __floordiv__(self, o):
        return constexpr(self.value // self._unwrap(o))

    def __eq__(self, o):
        return self.value == self._unwrap(o)

    def __lt__(self, o):
        return self.value < self._unwrap(o)

    def __le__(self, o):
        return self.value <= self._unwrap(o)

    def __hash__(self):
        return hash(self.value)

    def __index__(self):
        return int(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f"constexpr[{self.value!r}]"


def _device_only(fn):
    name = fn.__name__

    def stub(*args, **kwargs):
        raise RuntimeError(
            f"nl.{name} can only be called inside a @newt.jit kernel "
            f"(or via the interpreter with NEWT_INTERPRET=1)"
        )

    stub.__name__ = name
    stub.__doc__ = fn.__doc__
    stub._newt_builtin = name
    return stub


# -- core -------------------------------------------------------------------

@_device_only
def program_id(axis):
    """Index of the current program instance along the given grid axis (0-2)."""

@_device_only
def num_programs(axis):
    """Number of program instances along the given grid axis."""

@_device_only
def arange(start, end):
    """1D int32 block [start, end). end-start must be a power of two."""

@_device_only
def zeros(shape, dtype=float32):
    """Block of zeros. Each dim must be a power of two."""

@_device_only
def full(shape, value, dtype=float32):
    """Block filled with a scalar value."""

@_device_only
def load(pointer, mask=None, other=None):
    """Load a block from memory. Masked-off lanes yield `other` (default 0)."""

@_device_only
def store(pointer, value, mask=None):
    """Store a block to memory."""

# -- shape manipulation -----------------------------------------------------

@_device_only
def expand_dims(x, axis):
    """Insert a size-1 dimension (same as x[:, None])."""

@_device_only
def broadcast_to(x, shape):
    """Broadcast a block to a larger shape."""

@_device_only
def reshape(x, shape):
    """Reshape, preserving the total number of elements (row-major)."""

@_device_only
def trans(x):
    """Transpose a 2D block."""

@_device_only
def cast(x, dtype):
    """Elementwise cast (same as x.to(dtype))."""

# -- reductions -------------------------------------------------------------

@_device_only
def sum(x, axis=None):
    """Sum along an axis (or all axes). fp16/bf16 reduce in fp32."""

@_device_only
def max(x, axis=None):
    """Maximum along an axis (or all axes)."""

@_device_only
def min(x, axis=None):
    """Minimum along an axis (or all axes)."""

# -- linear algebra ---------------------------------------------------------

@_device_only
def dot(a, b, acc=None, out_dtype=float32):
    """Block matmul a[M,K] @ b[K,N] -> [M,N] using tensor cores (WMMA).

    fp16/bf16 inputs use hmma; fp32 inputs use tf32 (like Triton's default).
    Pass `acc` to accumulate in registers across loop iterations.
    M, N must be multiples of 16; K a multiple of 16 (8 for tf32).
    """

# -- elementwise math -------------------------------------------------------

@_device_only
def exp(x): ...

@_device_only
def exp2(x): ...

@_device_only
def log(x): ...

@_device_only
def log2(x): ...

@_device_only
def sqrt(x): ...

@_device_only
def rsqrt(x): ...

@_device_only
def sin(x): ...

@_device_only
def cos(x): ...

@_device_only
def tanh(x): ...

@_device_only
def erf(x): ...

@_device_only
def sigmoid(x): ...

@_device_only
def abs(x): ...

@_device_only
def floor(x): ...

@_device_only
def ceil(x): ...

@_device_only
def maximum(a, b): ...

@_device_only
def minimum(a, b): ...

@_device_only
def where(cond, a, b):
    """Elementwise select."""

@_device_only
def fma(a, b, c):
    """a * b + c, fused."""

# -- atomics ----------------------------------------------------------------

@_device_only
def atomic_add(pointer, value, mask=None): ...

@_device_only
def atomic_max(pointer, value, mask=None): ...

# -- misc -------------------------------------------------------------------

@_device_only
def cdiv(a, b):
    """Ceiling division."""

@_device_only
def multiple_of(x, value):
    """Compiler hint (accepted for Triton compatibility; currently a no-op)."""

@_device_only
def max_contiguous(x, value):
    """Compiler hint (accepted for Triton compatibility; currently a no-op)."""

@_device_only
def static_assert(cond, msg=""):
    """Compile-time assertion on constexpr values."""

@_device_only
def static_print(*args):
    """Print constexpr values at compile time."""
