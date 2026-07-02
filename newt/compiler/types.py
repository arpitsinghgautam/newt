"""Type system: scalar dtypes, pointer types, and block (tile) types."""


class dtype:
    def __init__(self, name, ctype, itemsize, kind, torch_name=None):
        self.name = name          # newt-facing name, e.g. "float32"
        self.ctype = ctype        # CUDA C type, e.g. "float"
        self.itemsize = itemsize
        self.kind = kind          # 'f' float, 'i' signed int, 'u' unsigned, 'b' bool
        self.torch_name = torch_name or name

    @property
    def is_floating(self):
        return self.kind == "f"

    @property
    def is_int(self):
        return self.kind in ("i", "u", "b")

    def __repr__(self):
        return f"nl.{self.name}"

    def __eq__(self, other):
        return isinstance(other, dtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


float64 = dtype("float64", "double", 8, "f")
float32 = dtype("float32", "float", 4, "f")
float16 = dtype("float16", "__half", 2, "f")
bfloat16 = dtype("bfloat16", "__nv_bfloat16", 2, "f")
int64 = dtype("int64", "long long", 8, "i")
int32 = dtype("int32", "int", 4, "i")
int16 = dtype("int16", "short", 2, "i")
int8 = dtype("int8", "signed char", 1, "i")
uint32 = dtype("uint32", "unsigned int", 4, "u")
uint8 = dtype("uint8", "unsigned char", 1, "u")
int1 = dtype("int1", "bool", 1, "b")  # masks

ALL_DTYPES = [float64, float32, float16, bfloat16, int64, int32, int16, int8, uint32, uint8, int1]
_BY_NAME = {d.name: d for d in ALL_DTYPES}

TORCH_TO_NEWT = {
    "torch.uint32": uint32,
    "torch.float32": float32,
    "torch.float64": float64,
    "torch.float16": float16,
    "torch.bfloat16": bfloat16,
    "torch.int64": int64,
    "torch.int32": int32,
    "torch.int16": int16,
    "torch.int8": int8,
    "torch.uint8": uint8,
    "torch.bool": int1,
}


class pointer_type:
    def __init__(self, element: dtype):
        self.element = element
        self.element_ty = element  # Triton-compatible alias (ptr.dtype.element_ty)
        self.name = f"pointer<{element.name}>"

    @property
    def ctype(self):
        return f"{self.element.ctype}*"

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, pointer_type) and self.element == other.element

    def __hash__(self):
        return hash(("ptr", self.element))


# fp16/bf16 compute in registers is done natively; reductions and math
# functions promote through float32 for accuracy.
_PROMOTE_ORDER = ["int1", "int8", "uint8", "int16", "int32", "uint32", "int64",
                  "float16", "bfloat16", "float32", "float64"]


def promote(a: dtype, b: dtype) -> dtype:
    """Numpy-flavored promotion, simplified: floats beat ints, wider beats narrower."""
    if a == b:
        return a
    if a.is_floating and not b.is_floating:
        return a
    if b.is_floating and not a.is_floating:
        return b
    if a.is_floating and b.is_floating:
        # fp16 + bf16 -> float32 (no implicit ordering between the two)
        if {a.name, b.name} == {"float16", "bfloat16"}:
            return float32
        return a if _PROMOTE_ORDER.index(a.name) >= _PROMOTE_ORDER.index(b.name) else b
    return a if _PROMOTE_ORDER.index(a.name) >= _PROMOTE_ORDER.index(b.name) else b


def broadcast_shapes(s1, s2):
    """Numpy broadcasting rules on compile-time shapes."""
    if s1 == ():
        return s2
    if s2 == ():
        return s1
    ndim = max(len(s1), len(s2))
    a = (1,) * (ndim - len(s1)) + tuple(s1)
    b = (1,) * (ndim - len(s2)) + tuple(s2)
    out = []
    for x, y in zip(a, b):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise TypeError(f"incompatible block shapes {s1} and {s2}")
    return tuple(out)
