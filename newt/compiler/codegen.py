"""The newt compiler: Python AST -> CUDA C++.

Execution model (same as Triton):
  * one "program" = one CUDA thread block with num_warps*32 threads
  * block tensors live in registers, distributed cyclically across threads:
    element with row-major linear index i is held by thread (i % T) in
    register slot (i / T). Contiguous loads/stores are perfectly coalesced.
  * reductions: register partials -> warp shuffles -> shared memory
  * broadcasts between different-sized blocks go through shared memory
  * nl.dot stages operands in shared memory and uses WMMA tensor-core
    fragments (fp16/bf16 -> hmma, fp32 -> tf32), with the accumulator kept
    in fragment registers across loop iterations
"""

import ast
import math
import os

from . import types as tp


class CompileError(Exception):
    pass


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

CONSTEXPR = "constexpr"  # compile-time python value (int/float/bool/tuple/dtype/...)
UNIFORM = "uniform"      # per-thread C scalar, same value in every thread
CYCLIC = "cyclic"        # register array, cyclic distribution across threads
FRAG = "frag"            # WMMA accumulator fragments (from nl.dot)
LAZY_ZERO = "lazy_zero"  # nl.zeros not yet materialized (layout decided by use)
LAZY_LOAD = "lazy_load"  # nl.load not yet emitted; a dot can cp.async it to smem


class Value:
    def __init__(self, layout, dtype=None, shape=(), var=None, base=None,
                 pyval=None, meta=None):
        self.layout = layout
        self.dtype = dtype        # tp.dtype or tp.pointer_type
        self.shape = tuple(shape)
        self.var = var            # C identifier (array name for blocks)
        self.base = base          # C pointer expression, for pointer blocks
        self.pyval = pyval        # for constexpr
        self.meta = meta          # for FRAG: dict(M,N,WM,WN,FM,FN,KF) ; for LAZY: ph index

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def is_ptr(self):
        return isinstance(self.dtype, tp.pointer_type)

    def __repr__(self):
        return f"Value({self.layout}, {self.dtype}, {self.shape}, {self.var or self.pyval})"


def cx(pyval):
    return Value(CONSTEXPR, pyval=pyval)


# ---------------------------------------------------------------------------
# Pre-pass: find loop-carried / hoisted names and dot accumulators
# ---------------------------------------------------------------------------


def _assigned_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(n.target, ast.Name):
                out.add(n.target.id)
        elif isinstance(n, ast.For):
            if isinstance(n.target, ast.Name):
                out.add(n.target.id)
    return out


def _read_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


class _Prepass:
    """Computes:
    stable: names needing persistent C storage (loop-carried or branch-defined)
    hoist:  {stmt-node-id: names first defined inside it but read after it}
    dot_accs: names that flow into nl.dot as the accumulator
    """

    def __init__(self, fndef: ast.FunctionDef):
        self.stable = set()
        self.hoist = {}
        self.dot_accs = set()
        self._scan_body(fndef.body, assigned_before=set())
        self._scan_dots(fndef)

    def _scan_body(self, body, assigned_before):
        seen = set(assigned_before)
        for i, stmt in enumerate(body):
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                inner = _assigned_names(stmt)
                # loop-carried: assigned both before and inside the loop
                # (same-iteration def-then-use needs no persistent storage)
                carried = inner & seen
                self.stable |= carried
                # names born inside but read after the statement
                after_reads = set()
                for later in body[i + 1:]:
                    after_reads |= _read_names(later)
                hoisted = (inner - seen) & after_reads
                if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                    hoisted.discard(stmt.target.id)
                self.hoist[id(stmt)] = hoisted
                self.stable |= hoisted
                # recurse for nested statements
                bodies = [stmt.body] + ([stmt.orelse] if stmt.orelse else [])
                for b in bodies:
                    self._scan_body(b, seen | inner)
                seen |= inner
            else:
                seen |= _assigned_names(stmt)

    def _scan_dots(self, fndef):
        for n in ast.walk(fndef):
            if isinstance(n, ast.Call):
                fname = None
                if isinstance(n.func, ast.Attribute):
                    fname = n.func.attr
                elif isinstance(n.func, ast.Name):
                    fname = n.func.id
                if fname == "dot":
                    if len(n.args) >= 3 and isinstance(n.args[2], ast.Name):
                        self.dot_accs.add(n.args[2].id)
                    for kw in n.keywords:
                        if kw.arg == "acc" and isinstance(kw.value, ast.Name):
                            self.dot_accs.add(kw.value.id)
            # acc = acc + nl.dot(...) / acc += nl.dot(...)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                if any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "dot"
                       for c in ast.walk(n.value)):
                    self.dot_accs.add(n.target.id)
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                tname = n.targets[0].id
                has_dot = any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "dot"
                              for c in ast.walk(n.value))
                if has_dot and tname in _read_names(n.value):
                    self.dot_accs.add(tname)


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

_CKEYWORDS = {
    "auto", "bool", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "int", "long",
    "register", "return", "short", "signed", "sizeof", "static", "struct", "switch",
    "template", "this", "typedef", "union", "unsigned", "void", "volatile", "while",
}

_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.BitAnd: "&", ast.BitOr: "|",
    ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}

_MATH_FNS = {
    # name -> (f32 fn, f64 fn)
    "exp": ("expf", "exp"), "exp2": ("exp2f", "exp2"), "log": ("logf", "log"),
    "log2": ("log2f", "log2"), "sqrt": ("sqrtf", "sqrt"), "rsqrt": ("rsqrtf", "rsqrt"),
    "sin": ("sinf", "sin"), "cos": ("cosf", "cos"), "tanh": ("tanhf", "tanh"),
    "erf": ("erff", "erf"), "floor": ("floorf", "floor"), "ceil": ("ceilf", "ceil"),
}

MAX_SMEM = 96 * 1024


class Codegen(ast.NodeVisitor):
    def __init__(self, fndef, fn_globals, param_kinds, constexprs, num_warps, kernel_name):
        """param_kinds: list of (name, Value) for runtime params (already typed);
        constexprs: dict name -> python value."""
        self.fndef = fndef
        self.globals = fn_globals
        self.num_warps = num_warps
        self.T = num_warps * 32
        self.name = kernel_name
        self.lines = []
        self.indent = 1
        self.tmpc = 0
        self.smem_bytes = 0
        self.uses = set()  # 'fp16', 'bf16', 'mma'
        self.pre = _Prepass(fndef)
        self.locals = {}
        self.storage = {}  # stable name -> Value (persistent storage)
        for name, v in param_kinds:
            self.locals[name] = v
        for name, v in constexprs.items():
            self.locals[name] = cx(v)
        # vector width (elements per group in the group-cyclic layout):
        # sized so the widest-need pointer gets 16-byte transactions
        vec = 1
        for _, v in param_kinds:
            if isinstance(v.dtype, tp.pointer_type):
                vec = max(vec, 16 // v.dtype.element.itemsize)
        self.VEC = min(vec, 8)
        env = os.environ.get("NEWT_VEC")
        if env:
            self.VEC = int(env)
        # cross-iteration dot pipeline state:
        # ring_reservations: dedicated smem regions beyond the scratch arena
        # frag_pending: acc var -> parameters of a deferred (not yet run) mma
        self.ring_reservations = []
        self.frag_pending = {}
        self._prologue_decls = []

    # -- infrastructure ------------------------------------------------------

    def err(self, node, msg):
        line = getattr(node, "lineno", "?")
        raise CompileError(f"{self.fndef.name}:{line}: {msg}")

    def emit(self, s):
        self.lines.append("  " * self.indent + s)

    def fresh(self, prefix="t"):
        self.tmpc += 1
        return f"_{prefix}{self.tmpc}"

    def track_smem(self, nbytes):
        nbytes = (nbytes + 15) // 16 * 16
        if nbytes > MAX_SMEM:
            raise CompileError(
                f"operation needs {nbytes} bytes of shared memory (max {MAX_SMEM}); "
                f"use smaller block sizes")
        self.smem_bytes = max(self.smem_bytes, nbytes)

    def slots(self, numel):
        """Per-thread register array length for a block of `numel` elements.

        Group-cyclic layout: element with linear index i belongs to thread
        (i // VEC) % T, register slot (i // (T*VEC)) * VEC + i % VEC. Each
        thread owns groups of VEC consecutive elements so global accesses can
        use 16-byte vector transactions.
        """
        chunk = self.T * self.VEC
        groups = max(1, numel // chunk) if numel >= chunk else 1
        return groups * self.VEC

    def guard(self, numel):
        """Validity condition (in terms of _s/_tid), or None if all valid."""
        chunk = self.T * self.VEC
        if numel >= chunk:
            if numel % chunk != 0:
                raise CompileError(f"block numel {numel} not a multiple of {chunk}")
            return None
        return f"({self.lin_expr()} < {numel})"

    def slot_loop(self, numel):
        """Emit per-thread element loop; returns guard or None. Close with end_loop()."""
        S = self.slots(numel)
        self.emit("#pragma unroll")
        self.emit(f"for (int _s = 0; _s < {S}; ++_s) {{")
        self.indent += 1
        return self.guard(numel)

    def end_loop(self):
        self.indent -= 1
        self.emit("}")

    def lin_expr(self):
        """Global linear index of per-thread element _s (group-cyclic layout)."""
        V, T = self.VEC, self.T
        if V == 1:
            return f"(_s * {T} + _tid)"
        return f"(((_s / {V}) * {T} + _tid) * {V} + (_s % {V}))"

    # -- literals & casts ----------------------------------------------------

    def literal(self, pyval, dtype):
        if dtype is None:
            dtype = tp.float32 if isinstance(pyval, float) else tp.int32
        if isinstance(pyval, bool):
            pyval = pyval if dtype == tp.int1 else int(pyval)
        if dtype == tp.int1:
            return "true" if pyval else "false"
        if dtype.kind in ("i", "u"):
            v = int(pyval)
            if v == -2147483648:
                return "(-2147483647 - 1)"
            suffix = "LL" if dtype == tp.int64 else ""
            return f"{v}{suffix}"
        # floats
        f = float(pyval)
        if math.isinf(f):
            bits = "0x7f800000" if f > 0 else "0xff800000"
            core = f"__int_as_float({bits})"
        elif math.isnan(f):
            core = "__int_as_float(0x7fc00000)"
        else:
            core = f"{f!r}f" if dtype != tp.float64 else f"{f!r}"
        if dtype == tp.float16:
            self.uses.add("fp16")
            return f"__float2half({core})"
        if dtype == tp.bfloat16:
            self.uses.add("bf16")
            return f"__float2bfloat16({core})"
        return core

    def convert(self, expr, src, dst):
        if src == dst:
            return expr
        if src == tp.float16:
            self.uses.add("fp16")
            f = f"__half2float({expr})"
            return f if dst == tp.float32 else self.convert(f, tp.float32, dst)
        if src == tp.bfloat16:
            self.uses.add("bf16")
            f = f"__bfloat162float({expr})"
            return f if dst == tp.float32 else self.convert(f, tp.float32, dst)
        if dst == tp.float16:
            self.uses.add("fp16")
            return f"__float2half({self.convert(expr, src, tp.float32)})"
        if dst == tp.bfloat16:
            self.uses.add("bf16")
            return f"__float2bfloat16({self.convert(expr, src, tp.float32)})"
        return f"(({dst.ctype})({expr}))"

    def cdtype(self, v: Value):
        """Effective dtype of a constexpr/typed value."""
        if v.layout == CONSTEXPR:
            if isinstance(v.pyval, bool):
                return tp.int1
            if isinstance(v.pyval, int):
                return tp.int64 if not (-2**31 <= v.pyval < 2**31) else tp.int32
            if isinstance(v.pyval, float):
                return tp.float32
            raise CompileError(f"constexpr {v.pyval!r} has no device type")
        return v.dtype

    # -- broadcasting / materialization ---------------------------------------

    def materialize(self, v: Value, want=CYCLIC, shape=None):
        """Ensure v is a device value (not lazy); optionally to a given layout."""
        if v.layout == LAZY_ZERO:
            self._materialize_lazy(v, want, shape)
        if v.layout == LAZY_LOAD:
            self._materialize_load(v)
        if v.layout == FRAG and want == CYCLIC:
            v = self.frag_to_cyclic(v)
        return v

    def flush_lazy_loads(self):
        """Materialize deferred loads. Called before anything that could
        mutate memory or pointer state they depend on (stores, atomics,
        loop/branch entry, loop-carried rebinds). Loads already consumed by
        a dot are dropped (poisoned: their pointers may be stale later)."""
        for v in list(getattr(self, "pending_lazy", [])):
            if v.layout == LAZY_LOAD:
                if v.meta.get("consumed"):
                    v.meta["poisoned"] = True
                else:
                    self._materialize_load(v)
        self.pending_lazy = []

    def _materialize_lazy(self, v, want, shape):
        ph = v.meta["ph"]
        if want == FRAG:
            m = v.meta["frag_meta"]
            decl = self._frag_decl_and_fill(v.var, m)
            self.lines[ph] = decl
            v.layout = FRAG
            v.meta = m
        else:
            S = self.slots(v.numel)
            ct = v.dtype.ctype
            zero = self.literal(0, v.dtype)
            pad = "  " * 1
            self.lines[ph] = (
                f"{pad}{ct} {v.var}[{S}];\n"
                f"{pad}#pragma unroll\n"
                f"{pad}for (int _z = 0; _z < {S}; ++_z) {v.var}[_z] = {zero};"
            )
            v.layout = CYCLIC
            v.meta = None

    def broadcast_values(self, node, vals):
        """Broadcast a list of Values to a common shape; returns (vals, shape)."""
        shape = ()
        for v in vals:
            if v.layout in (CYCLIC, FRAG, LAZY_ZERO):
                shape = tp.broadcast_shapes(shape, v.shape)
        out = []
        for v in vals:
            if v.layout in (CONSTEXPR, UNIFORM):
                out.append(v)
            else:
                out.append(self.broadcast_to(node, v, shape))
        return out, shape

    def broadcast_to(self, node, v, shape):
        if v.layout in (UNIFORM, CONSTEXPR):
            # scalar -> block: plain fill, no smem staging
            d = self.cdtype(v)
            out = self.fresh("bf")
            rnumel = math.prod(shape) if shape else 1
            S = self.slots(rnumel)
            src = self.literal(v.pyval, d) if v.layout == CONSTEXPR else v.var
            self.emit(f"{d.ctype} {out}[{S}];")
            self.emit("#pragma unroll")
            self.emit(f"for (int _c = 0; _c < {S}; ++_c) {out}[_c] = {src};")
            return Value(CYCLIC, d, shape, out)
        v = self.materialize(v)
        if v.shape == tuple(shape):
            return v
        # numel-preserving broadcasts don't move data (row-major linearization
        # is unchanged when only size-1 dims are inserted/aligned)
        rnumel = 1
        for d in shape:
            rnumel *= d
        if v.numel == rnumel:
            return Value(v.layout, v.dtype, shape, v.var, base=v.base)
        # real broadcast: stage source in shared memory, gather
        tp.broadcast_shapes(v.shape, shape)  # validate
        is_ptr = v.is_ptr
        ct = ("long long" if False else "int") if is_ptr else v.dtype.ctype
        if is_ptr:
            # offsets array is int32
            ct = "int"
            elem_bytes = 4
        else:
            elem_bytes = v.dtype.itemsize
        self.track_smem(v.numel * elem_bytes)
        buf = self.fresh("bb")
        self.emit("__syncthreads();")
        self.emit(f"{ct}* {buf} = ({ct}*)_smem;")
        g = self.slot_loop(v.numel)
        w = f"{buf}[{self.lin_expr()}] = {v.var}[_s];"
        self.emit(f"if ({g}) {w}" if g else w)
        self.end_loop()
        self.emit("__syncthreads();")
        out = self.fresh("bc")
        S = self.slots(rnumel)
        self.emit(f"{ct} {out}[{S}];")
        g = self.slot_loop(rnumel)
        self.emit(f"int _j = {self.lin_expr()};")
        # decompose _j in result shape; build source linear index
        src_shape = (1,) * (len(shape) - len(v.shape)) + v.shape
        sstrides = [1] * len(src_shape)
        for d in range(len(src_shape) - 2, -1, -1):
            sstrides[d] = sstrides[d + 1] * src_shape[d + 1]
        rstrides = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            rstrides[d] = rstrides[d + 1] * shape[d + 1]
        terms = []
        for d in range(len(shape)):
            if src_shape[d] > 1:
                coord = f"((_j / {rstrides[d]}) % {shape[d]})"
                terms.append(f"{coord} * {sstrides[d]}" if sstrides[d] != 1 else coord)
        src_idx = " + ".join(terms) if terms else "0"
        r = f"{out}[_s] = {buf}[{src_idx}];"
        self.emit(f"if ({g}) {r}" if g else r)
        self.end_loop()
        return Value(CYCLIC, v.dtype, shape, out, base=v.base)

    def slot_expr(self, v: Value, dtype=None):
        """C expression for the current slot's element inside a slot loop."""
        d = dtype or self.cdtype(v)
        if v.layout == CONSTEXPR:
            return self.literal(v.pyval, d)
        if v.layout == UNIFORM:
            return self.convert(v.var, v.dtype, d) if dtype else v.var
        if v.layout == CYCLIC:
            e = f"{v.var}[_s]"
            return self.convert(e, v.dtype, d) if dtype else e
        raise CompileError(f"cannot index layout {v.layout}")

    # -- elementwise ----------------------------------------------------------

    def elementwise(self, node, operands, rdtype, expr_fn):
        """Apply expr_fn over broadcast operands. expr_fn(list-of-C-exprs)->C expr."""
        operands = [self.materialize(o) if o.layout in (LAZY_ZERO, LAZY_LOAD) else o
                    for o in operands]
        # frag fast path: frag op scalar/constexpr stays in fragments
        frags = [o for o in operands if o.layout == FRAG]
        if frags:
            others = [o for o in operands if o.layout not in (FRAG, CONSTEXPR, UNIFORM)]
            if not others and len(frags) <= 2 and all(
                f.meta == frags[0].meta or f.meta is None for f in frags
            ):
                return self._frag_elementwise(node, operands, rdtype, expr_fn)
            operands = [self.frag_to_cyclic(o) if o.layout == FRAG else o for o in operands]
        operands, shape = self.broadcast_values(node, operands)
        if shape == ():  # all scalar
            exprs = [self.slot_expr_scalar(o, rdtype) for o in operands]
            t = self.fresh()
            self.emit(f"{rdtype.ctype} {t} = {expr_fn(exprs)};")
            return Value(UNIFORM, rdtype, (), t)
        out = self.fresh()
        S = self.slots(max(1, math.prod(shape)))
        self.emit(f"{rdtype.ctype} {out}[{S}];")
        g = self.slot_loop(math.prod(shape))
        exprs = [self.slot_expr(o, None) for o in operands]
        line = f"{out}[_s] = {expr_fn(exprs)};"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, rdtype, shape, out)

    def slot_expr_scalar(self, v, dtype=None):
        if v.layout == CONSTEXPR:
            return self.literal(v.pyval, dtype or self.cdtype(v))
        return v.var

    def _frag_elementwise(self, node, operands, rdtype, expr_fn):
        for o in operands:
            if o.layout == FRAG and o.var in getattr(self, "frag_pending", {}):
                self._emit_flush(o.var)
        ref = next(o for o in operands if o.layout == FRAG)
        m = ref.meta
        out = self.fresh("fe")
        self.emit(self._frag_decl(out, m))
        self.emit("#pragma unroll")
        self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fe = 0; _fe < {out}[_fm][_fn].num_elements; ++_fe) {{")
        self.indent += 1
        exprs = []
        for o in operands:
            if o.layout == FRAG:
                exprs.append(f"{o.var}[_fm][_fn].x[_fe]")
            elif o.layout == CONSTEXPR:
                exprs.append(self.literal(o.pyval, tp.float32))
            else:
                exprs.append(self.convert(o.var, o.dtype, tp.float32))
        self.emit(f"{out}[_fm][_fn].x[_fe] = {expr_fn(exprs)};")
        self.end_loop()
        return Value(FRAG, tp.float32, ref.shape, out, meta=m)

    # -- fragments ------------------------------------------------------------

    def _frag_meta(self, M, N, KF):
        """Choose the warp tiling (WM x WN) for an M x N accumulator.

        If the fragment grid is smaller than num_warps, use fewer warps and
        leave the rest idle (guarded by _warp < W in the collective sections).
        """
        W = self.num_warps
        while W >= 1:
            best = None
            for WM in [1, 2, 4, 8, 16, 32]:
                if WM > W or W % WM != 0:
                    continue
                WN = W // WM
                if (M // 16) % WM == 0 and (N // 16) % WN == 0:
                    FM, FN = M // 16 // WM, N // 16 // WN
                    score = abs(FM - FN)
                    if best is None or score < best[0]:
                        best = (score, WM, WN, FM, FN)
            if best is not None:
                _, WM, WN, FM, FN = best
                return dict(M=M, N=N, WM=WM, WN=WN, FM=FM, FN=FN, KF=KF, W=WM * WN)
            W //= 2
        raise CompileError(
            f"cannot tile {M}x{N} dot output; M and N must be multiples of 16")

    def _frag_decl(self, var, m):
        self.uses.add("mma")
        return (f"wmma::fragment<wmma::accumulator, 16, 16, {m['KF']}, float> "
                f"{var}[{m['FM']}][{m['FN']}];")

    def _frag_decl_and_fill(self, var, m):
        pad = "  "
        return (
            pad + self._frag_decl(var, m) + "\n"
            + pad + "#pragma unroll\n"
            + pad + f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)\n"
            + pad + "#pragma unroll\n"
            + pad + f"for (int _fn = 0; _fn < {m['FN']}; ++_fn) "
            + f"wmma::fill_fragment({var}[_fm][_fn], 0.0f);"
        )

    def frag_to_cyclic(self, v):
        """Store accumulator fragments to smem, read back into cyclic registers.

        Done band-by-band (16 rows at a time) so the scratch buffer stays at
        16*(N+pad) floats instead of M*(N+pad) - the full buffer would
        dominate the kernel's smem footprint and wreck occupancy.
        """
        if v.var in getattr(self, "frag_pending", {}):
            self._emit_flush(v.var)
        m = v.meta
        M, N = m["M"], m["N"]
        PC = 8
        ld = N + PC
        banded = (16 * N) % (self.T * self.VEC) == 0 and M > 16
        out = self.fresh()
        S = self.slots(M * N)
        self.emit(f"float {out}[{S}];")
        buf = self.fresh("cs")
        if banded:
            slots_per_band = (16 * N) // self.T
            self.track_smem(16 * ld * 4)
            self.emit(f"float* {buf} = (float*)_smem;")
            self.emit("{")
            self.indent += 1
            self.emit(f"int _wm = _warp / {m['WN']}, _wn = _warp % {m['WN']};")
            self.emit(f"for (int _bi = 0; _bi < {M // 16}; ++_bi) {{")
            self.indent += 1
            self.emit("__syncthreads();")
            # the warp owning fragment-row _bi stores its FN fragments
            self.emit(f"if (_wm == _bi / {m['FM']}) {{")
            self.indent += 1
            self.emit(f"int _fm = _bi % {m['FM']};")
            self.emit("#pragma unroll")
            self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
            self.emit(f"  wmma::store_matrix_sync(&{buf}[(_wn * {m['FN']} + _fn) * 16], "
                      f"{v.var}[_fm][_fn], {ld}, wmma::mem_row_major);")
            self.end_loop()
            self.emit("__syncthreads();")
            self.emit("#pragma unroll")
            self.emit(f"for (int _bs = 0; _bs < {slots_per_band}; ++_bs) {{")
            self.indent += 1
            self.emit(f"int _s = _bi * {slots_per_band} + _bs;")
            self.emit(f"int _j = {self.lin_expr()};")
            self.emit(f"{out}[_s] = {buf}[(_j / {N} - _bi * 16) * {ld} + (_j % {N})];")
            self.end_loop()
            self.end_loop()
            self.end_loop()
            return Value(CYCLIC, tp.float32, v.shape, out)
        self.track_smem(M * ld * 4)
        self.emit("__syncthreads();")
        self.emit(f"float* {buf} = (float*)_smem;")
        self.emit(f"if (_warp < {m['W']}) {{")
        self.indent += 1
        self.emit(f"int _wm = _warp / {m['WN']}, _wn = _warp % {m['WN']};")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
        self.emit(f"  wmma::store_matrix_sync(&{buf}[(_wm * {m['FM']} + _fm) * 16 * {ld} + "
                  f"(_wn * {m['FN']} + _fn) * 16], {v.var}[_fm][_fn], {ld}, wmma::mem_row_major);")
        self.end_loop()
        self.emit("__syncthreads();")
        g = self.slot_loop(M * N)
        self.emit(f"int _j = {self.lin_expr()};")
        line = f"{out}[_s] = {buf}[(_j / {N}) * {ld} + (_j % {N})];"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, tp.float32, v.shape, out)

    def cyclic_to_frag(self, v, m):
        """Load a cyclic fp32 [M,N] block into WMMA accumulator fragments.

        Needed when a dot accumulator is rescaled elementwise between dot
        calls (flash-attention's online softmax). Banded like frag_to_cyclic.
        """
        M, N = m["M"], m["N"]
        PC = 8
        ld = N + PC
        out = self.fresh("cf")
        self.emit(self._frag_decl(out, m))
        buf = self.fresh("cs")
        banded = (16 * N) % (self.T * self.VEC) == 0 and M > 16
        if banded:
            slots_per_band = (16 * N) // self.T
            self.track_smem(16 * ld * 4)
            self.emit(f"float* {buf} = (float*)_smem;")
            self.emit("{")
            self.indent += 1
            self.emit(f"int _wm = _warp / {m['WN']}, _wn = _warp % {m['WN']};")
            self.emit(f"for (int _bi = 0; _bi < {M // 16}; ++_bi) {{")
            self.indent += 1
            self.emit("__syncthreads();")
            self.emit("#pragma unroll")
            self.emit(f"for (int _bs = 0; _bs < {slots_per_band}; ++_bs) {{")
            self.indent += 1
            self.emit(f"int _s = _bi * {slots_per_band} + _bs;")
            self.emit(f"int _j = {self.lin_expr()};")
            self.emit(f"{buf}[(_j / {N} - _bi * 16) * {ld} + (_j % {N})] = {v.var}[_s];")
            self.end_loop()
            self.emit("__syncthreads();")
            self.emit(f"if (_wm == _bi / {m['FM']}) {{")
            self.indent += 1
            self.emit(f"int _fm = _bi % {m['FM']};")
            self.emit("#pragma unroll")
            self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
            self.emit(f"  wmma::load_matrix_sync({out}[_fm][_fn], "
                      f"&{buf}[(_wn * {m['FN']} + _fn) * 16], {ld}, wmma::mem_row_major);")
            self.end_loop()
            self.end_loop()
            self.end_loop()
            return Value(FRAG, tp.float32, v.shape, out, meta=m)
        self.track_smem(M * ld * 4)
        self.emit("__syncthreads();")
        self.emit(f"float* {buf} = (float*)_smem;")
        g = self.slot_loop(M * N)
        self.emit(f"int _j = {self.lin_expr()};")
        line = f"{buf}[(_j / {N}) * {ld} + (_j % {N})] = {v.var}[_s];"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        self.emit("__syncthreads();")
        self.emit(f"if (_warp < {m['W']}) {{")
        self.indent += 1
        self.emit(f"int _wm = _warp / {m['WN']}, _wn = _warp % {m['WN']};")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
        self.emit(f"  wmma::load_matrix_sync({out}[_fm][_fn], "
                  f"&{buf}[(_wm * {m['FM']} + _fm) * 16 * {ld} + (_wn * {m['FN']} + _fn) * 16], "
                  f"{ld}, wmma::mem_row_major);")
        self.end_loop()
        return Value(FRAG, tp.float32, v.shape, out, meta=m)

    # -- statements ------------------------------------------------------------

    def compile(self):
        self.emit("const int _tid = threadIdx.x;")
        self.emit("const int _lane = _tid & 31;")
        self.emit("const int _warp = _tid >> 5;")
        self.emit("(void)_lane; (void)_warp;")
        self._prologue_idx = len(self.lines)
        for stmt in self.fndef.body:
            self.visit(stmt)
        return self.assemble()

    def visit_Expr(self, node):
        # docstring or a call for side effects (store, atomics, asserts)
        if isinstance(node.value, ast.Constant):
            return
        self.visit_expr(node.value)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            self.err(node, "chained assignment not supported")
        tgt = node.targets[0]
        if isinstance(tgt, ast.Tuple):
            # a, b = expr1, expr2 - literal tuple RHS only
            if not (isinstance(node.value, ast.Tuple)
                    and len(node.value.elts) == len(tgt.elts)
                    and all(isinstance(t, ast.Name) for t in tgt.elts)):
                self.err(node, "tuple unpacking only supports `a, b = expr1, expr2` form")
            for t, v in zip(tgt.elts, node.value.elts):
                self.bind(node, t.id, self.visit_expr(v))
            return
        if not isinstance(tgt, ast.Name):
            self.err(node, "can only assign to plain names")
        val = self.visit_expr(node.value)
        self.bind(node, tgt.id, val)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return
        val = self.visit_expr(node.value)
        self.bind(node, node.target.id, val)

    def visit_AugAssign(self, node):
        cur = self.lookup(node, node.target.id)
        rhs = self.visit_expr(node.value)
        val = self.binop(node, type(node.op), cur, rhs)
        self.bind(node, node.target.id, val)

    def bind(self, node, name, val):
        if name not in self.pre.stable:
            # copy when aliasing mutable loop-carried storage: `prev = x`
            # must not observe later in-place updates of x
            if val.var is not None and any(
                s.var == val.var for s in self.storage.values() if s.var
            ):
                val = self._copy_value(val)
            self.locals[name] = val
            return
        if val.layout == LAZY_LOAD:
            val = self.materialize(val)
        in_hoist = any(name in h for h, _ in getattr(self, "_pending_hoist", []))
        store = self.storage.get(name)
        if store is None:
            if val.layout == LAZY_ZERO and not in_hoist:
                # keep lazy; storage is created on materialization
                self.locals[name] = val
                self.storage[name] = val
                return
            if val.layout == LAZY_ZERO:
                val = self.materialize(val, CYCLIC)
            store = self._declare_storage(name, val)
            self.storage[name] = store
            self.locals[name] = store
            self._copy_into(node, store, val)
            return
        if store.layout == LAZY_ZERO:
            # lazily-typed accumulator being reassigned before materialization
            if val is store:
                return
            self.materialize(store, CYCLIC)
        if val is store or (val.var is not None and val.var == store.var):
            return  # in-place update (e.g. dot accumulator)
        # rebinding persistent state (e.g. a_ptrs += ...) can invalidate
        # deferred loads that reference it
        self.flush_lazy_loads()
        if (val.layout, val.dtype, val.shape) != (store.layout, store.dtype, store.shape):
            val = self._coerce(node, val, store)
        self._copy_into(node, store, val)

    def _coerce(self, node, val, store):
        """Try to make val match the type of loop-carried storage."""
        if val.layout == LAZY_ZERO:
            want = store.layout if store.layout in (CYCLIC, FRAG) else CYCLIC
            val = self.materialize(val, want)
        if val.layout == FRAG and store.layout == CYCLIC:
            val = self.frag_to_cyclic(val)
        if val.layout == CYCLIC and val.shape != store.shape and val.numel == store.numel:
            val = Value(CYCLIC, val.dtype, store.shape, val.var)
        if val.layout == CYCLIC and val.dtype != store.dtype and val.shape == store.shape:
            val = self.cast(node, val, store.dtype)
        if val.layout == UNIFORM and store.layout == UNIFORM and val.dtype != store.dtype:
            if store.dtype.is_int and val.dtype.is_floating:
                self.err(node, f"loop-carried scalar was created as {store.dtype} but "
                               f"updated with {val.dtype}; initialize it as a float "
                               f"(e.g. `s = 0.0`)")
            t = self.fresh()
            self.emit(f"{store.dtype.ctype} {t} = {self.convert(val.var, val.dtype, store.dtype)};")
            val = Value(UNIFORM, store.dtype, (), t)
        if (val.layout, val.dtype, val.shape) != (store.layout, store.dtype, store.shape):
            self.err(node, f"loop-carried variable changed type: "
                           f"{store.layout}/{store.dtype}/{store.shape} -> "
                           f"{val.layout}/{val.dtype}/{val.shape}")
        return val

    def _copy_value(self, val):
        """Snapshot a value into fresh registers (breaks storage aliasing)."""
        if val.layout == UNIFORM:
            t = self.fresh()
            self.emit(f"{val.dtype.ctype} {t} = {val.var};")
            return Value(UNIFORM, val.dtype, (), t)
        if val.layout == CYCLIC:
            t = self.fresh("cp")
            S = self.slots(val.numel)
            ct = "int" if val.is_ptr else val.dtype.ctype
            self.emit(f"{ct} {t}[{S}];")
            self.emit("#pragma unroll")
            self.emit(f"for (int _c = 0; _c < {S}; ++_c) {t}[_c] = {val.var}[_c];")
            return Value(CYCLIC, val.dtype, val.shape, t, base=val.base)
        if val.layout == FRAG:
            if val.var in getattr(self, "frag_pending", {}):
                self._emit_flush(val.var)
            t = self.fresh("fcp")
            m = val.meta
            self.emit(self._frag_decl(t, m))
            self.emit("#pragma unroll")
            self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
            self.emit("#pragma unroll")
            self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn) "
                      f"{t}[_fm][_fn] = {val.var}[_fm][_fn];")
            return Value(FRAG, val.dtype, val.shape, t, meta=m)
        return val

    def _emit_decl(self, name, decl):
        """Emit a declaration, hoisting it above the enclosing loop/branch if
        the name is read after that statement (pre-pass hoist sets)."""
        for hoisted, pre_decl in getattr(self, "_pending_hoist", []):
            if name in hoisted:
                pre_decl[name] = decl
                return
        self.emit(decl)

    def _declare_storage(self, name, val):
        cname = f"v_{name}" if name not in _CKEYWORDS else f"v_{name}_"
        cname = f"{cname}_{self.tmpc}"
        self.tmpc += 1
        if val.layout == CONSTEXPR:
            # loop-carried scalar initialized from a constant (s = 0; s += ...)
            d = self.cdtype(val)
            self._emit_decl(name, f"{d.ctype} {cname};")
            return Value(UNIFORM, d, (), cname)
        if val.layout == UNIFORM:
            self._emit_decl(name, f"{val.dtype.ctype} {cname};")
            return Value(UNIFORM, val.dtype, (), cname)
        if val.layout == CYCLIC:
            S = self.slots(val.numel)
            ct = "int" if val.is_ptr else val.dtype.ctype
            self._emit_decl(name, f"{ct} {cname}[{S}];")
            if val.is_ptr:
                return Value(CYCLIC, val.dtype, val.shape, cname, base=val.base)
            return Value(CYCLIC, val.dtype, val.shape, cname)
        if val.layout == FRAG:
            self._emit_decl(name, self._frag_decl(cname, val.meta))
            return Value(FRAG, val.dtype, val.shape, cname, meta=val.meta)
        raise CompileError(f"cannot store layout {val.layout}")

    def _copy_into(self, node, store, val):
        if store.layout == UNIFORM:
            src = self.slot_expr_scalar(val) if val.layout == CONSTEXPR else val.var
            if val.layout == CONSTEXPR:
                src = self.literal(val.pyval, store.dtype)
            elif val.dtype != store.dtype:
                src = self.convert(src, val.dtype, store.dtype)
            self.emit(f"{store.var} = {src};")
        elif store.layout == CYCLIC:
            if val.layout == CONSTEXPR or val.layout == UNIFORM:
                lit = (self.literal(val.pyval, store.dtype) if val.layout == CONSTEXPR
                       else self.convert(val.var, val.dtype, store.dtype))
                self.slot_loop(store.numel)
                self.emit(f"{store.var}[_s] = {lit};")
                self.end_loop()
            else:
                if val.is_ptr and store.is_ptr and val.base != store.base:
                    self.err(node, "a loop-carried pointer block cannot switch its "
                                   "base tensor; use separate pointer variables")
                self.slot_loop(store.numel)
                self.emit(f"{store.var}[_s] = {val.var}[_s];")
                self.end_loop()
        elif store.layout == FRAG:
            if val.var in getattr(self, "frag_pending", {}):
                self._emit_flush(val.var)
            if store.var in getattr(self, "frag_pending", {}):
                # a deferred mma into the value being overwritten must retire
                # first, or it would later fire into the NEW contents
                self._emit_flush(store.var)
            m = store.meta
            self.emit("#pragma unroll")
            self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
            self.emit("#pragma unroll")
            self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn) "
                      f"{store.var}[_fm][_fn] = {val.var}[_fm][_fn];")

    def visit_For(self, node):
        self.flush_lazy_loads()
        if node.orelse:
            self.err(node, "for/else not supported")
        if not (isinstance(node.iter, ast.Call) and self._callee_name(node.iter.func) == "range"):
            self.err(node, "only `for i in range(...)` loops are supported")
        args = [self.visit_expr(a) for a in node.iter.args]
        lo, hi, step = cx(0), None, cx(1)
        if len(args) == 1:
            hi = args[0]
        elif len(args) == 2:
            lo, hi = args
        elif len(args) == 3:
            lo, hi, step = args
        else:
            self.err(node, "range() takes 1-3 arguments")

        def as_c(v):
            if v.layout == CONSTEXPR:
                return str(int(v.pyval))
            if v.layout == UNIFORM:
                return v.var
            self.err(node, "range bounds must be scalars")

        if not isinstance(node.target, ast.Name):
            self.err(node, "loop target must be a name")
        ivar = self.fresh("i")
        for_idx = len(self.lines)
        all_const = all(a.layout == CONSTEXPR for a in (lo, hi, step))
        if all_const:
            self.emit("#pragma unroll")
        self.emit(f"for (int {ivar} = {as_c(lo)}; {ivar} < {as_c(hi)}; {ivar} += {as_c(step)}) {{")
        self.indent += 1
        self.locals[node.target.id] = Value(UNIFORM, tp.int32, (), ivar)
        self._visit_hoisted_body(node, for_idx)
        self.indent -= 1
        self.emit("}")

    def _visit_hoisted_body(self, node, insert_idx, bodies=None):
        """Visit body; hoist declarations of names born inside but used after."""
        hoisted = self.pre.hoist.get(id(node), set())
        pre_decl = {}
        if hoisted:
            self._pending_hoist = getattr(self, "_pending_hoist", [])
            self._pending_hoist.append((hoisted, pre_decl))
        for stmt in (bodies if bodies is not None else node.body):
            self.visit(stmt)
        if hoisted:
            self._pending_hoist.pop()
            decls = list(pre_decl.values())
            for d in reversed(decls):
                self.lines.insert(insert_idx, "  " * (self.indent - 1) + d)

    def visit_While(self, node):
        self.flush_lazy_loads()
        if node.orelse:
            self.err(node, "while/else not supported")
        while_idx = len(self.lines)
        self.emit("while (true) {")
        self.indent += 1
        cond = self.visit_expr(node.test)
        if cond.layout == CONSTEXPR:
            if not cond.pyval:
                # dead loop; drop emitted lines
                del self.lines[while_idx:]
                return
            pass
        else:
            if cond.layout != UNIFORM:
                self.err(node, "while condition must be a scalar")
            self.emit(f"if (!({cond.var})) break;")
        self._visit_hoisted_body(node, while_idx)
        self.indent -= 1
        self.emit("}")

    def visit_If(self, node):
        self.flush_lazy_loads()
        cond = self.visit_expr(node.test)
        if cond.layout == CONSTEXPR:
            body = node.body if cond.pyval else node.orelse
            for stmt in body:
                self.visit(stmt)
            return
        if cond.layout != UNIFORM:
            self.err(node, "if condition must be a scalar (use nl.where for blocks)")
        if_idx = len(self.lines)
        self.emit(f"if ({cond.var}) {{")
        self.indent += 1
        self._visit_hoisted_body(node, if_idx, bodies=node.body)
        self.indent -= 1
        if node.orelse:
            self.emit("} else {")
            self.indent += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent -= 1
        self.emit("}")

    def visit_Return(self, node):
        if node.value is not None:
            self.err(node, "kernels cannot return values")
        self.emit("return;")

    def visit_Break(self, node):
        self.emit("break;")

    def visit_Continue(self, node):
        self.emit("continue;")

    def visit_Pass(self, node):
        pass

    def visit_Assert(self, node):
        pass  # host-level asserts are ignored on device

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            self.err(node, f"unsupported statement: {type(node).__name__}")
        super().generic_visit(node)

    # -- expressions -----------------------------------------------------------

    def lookup(self, node, name):
        if name in self.locals:
            return self.locals[name]
        if name in self.globals:
            g = self.globals[name]
            if isinstance(g, (int, float, bool, str, tuple, tp.dtype)):
                return cx(g)
            return cx(g)  # modules / functions resolved by callers
        import builtins

        if hasattr(builtins, name):
            return cx(getattr(builtins, name))
        self.err(node, f"undefined name '{name}'")

    def visit_expr(self, node):
        m = getattr(self, f"expr_{type(node).__name__}", None)
        if m is None:
            self.err(node, f"unsupported expression: {type(node).__name__}")
        return m(node)

    def expr_Name(self, node):
        return self.lookup(node, node.id)

    def expr_Constant(self, node):
        return cx(node.value)

    def expr_Tuple(self, node):
        vals = [self.visit_expr(e) for e in node.elts]
        if all(v.layout == CONSTEXPR for v in vals):
            return cx(tuple(v.pyval for v in vals))
        self.err(node, "tuples of runtime values are not supported")

    def expr_List(self, node):
        return self.expr_Tuple(node)

    def expr_Attribute(self, node):
        # Value attributes
        if isinstance(node.value, ast.Name) and node.value.id in self.locals:
            v = self.locals[node.value.id]
            if isinstance(v, Value) and v.layout != CONSTEXPR:
                if node.attr == "shape":
                    return cx(tuple(v.shape))
                if node.attr == "dtype":
                    return cx(v.dtype)
        base = self.visit_expr(node.value)
        if base.layout == CONSTEXPR:
            try:
                return cx(getattr(base.pyval, node.attr))
            except AttributeError:
                self.err(node, f"no attribute {node.attr!r}")
        if node.attr == "shape":
            return cx(tuple(base.shape))
        if node.attr == "dtype":
            return cx(base.dtype)
        self.err(node, f"unsupported attribute access .{node.attr}")

    def expr_UnaryOp(self, node):
        v = self.visit_expr(node.operand)
        if v.layout == CONSTEXPR:
            import operator

            ops = {ast.USub: operator.neg, ast.UAdd: operator.pos,
                   ast.Invert: operator.invert, ast.Not: operator.not_}
            return cx(ops[type(node.op)](v.pyval))
        v = self.materialize(v)
        d = v.dtype
        if isinstance(node.op, ast.USub):
            if d in (tp.float16, tp.bfloat16):
                return self.elementwise(node, [v], d, lambda e: self.convert(
                    f"(-{self.convert(e[0], d, tp.float32)})", tp.float32, d))
            return self.elementwise(node, [v], d, lambda e: f"(-{e[0]})")
        if isinstance(node.op, ast.Not):
            return self.elementwise(node, [v], tp.int1, lambda e: f"(!{e[0]})")
        if isinstance(node.op, ast.Invert):
            return self.elementwise(node, [v], d, lambda e: f"(~{e[0]})")
        return v

    def expr_BinOp(self, node):
        a = self.visit_expr(node.left)
        b = self.visit_expr(node.right)
        return self.binop(node, type(node.op), a, b)

    def binop(self, node, op, a, b):
        if a.layout == CONSTEXPR and b.layout == CONSTEXPR:
            import operator

            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                   ast.Mod: operator.mod, ast.Pow: operator.pow,
                   ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
                   ast.BitXor: operator.xor, ast.LShift: operator.lshift,
                   ast.RShift: operator.rshift}
            return cx(ops[op](a.pyval, b.pyval))
        # pointer arithmetic
        if (isinstance(a.dtype, tp.pointer_type) or isinstance(b.dtype, tp.pointer_type)):
            return self.ptr_arith(node, op, a, b)
        da, db = self.cdtype(a), self.cdtype(b)
        # python scalar constants adopt the typed operand's dtype (Triton's
        # rule: uint8_block + 200 wraps in uint8, fp16_block * 0.1 stays fp16);
        # float constants with integer blocks promote to fp32
        if a.layout == CONSTEXPR and b.layout != CONSTEXPR:
            rd = tp.promote(tp.float32, db) if (isinstance(a.pyval, float)
                                                and db.is_int) else db
        elif b.layout == CONSTEXPR and a.layout != CONSTEXPR:
            rd = tp.promote(da, tp.float32) if (isinstance(b.pyval, float)
                                                and da.is_int) else da
        else:
            rd = tp.promote(da, db)
        halfish = rd in (tp.float16, tp.bfloat16)
        cd = tp.float32 if halfish else rd  # compute dtype

        def wrap(fn):
            if not halfish:
                return fn
            return lambda es: self.convert(fn(es), tp.float32, rd)

        def conv_ops(es_needed_dtype):
            pass

        if op in _BINOPS:
            sym = _BINOPS[op]
            if sym in ("&", "|", "^", "<<", ">>") and rd.is_floating:
                self.err(node, f"bitwise {sym} on floats")
            fn = wrap(lambda es: f"({es[0]} {sym} {es[1]})")
            return self._emit_binop(node, a, b, rd, cd, fn)
        if op == ast.Div:
            if rd.is_int:
                fn = lambda es: f"({es[0]} / {es[1]})"
                return self._emit_binop(node, a, b, rd, cd, fn)
            fn = wrap(lambda es: f"({es[0]} / {es[1]})")
            return self._emit_binop(node, a, b, rd, cd, fn)
        if op == ast.FloorDiv:
            if rd.is_int:
                fn = lambda es: f"({es[0]} / {es[1]})"
            else:
                base = "floorf" if cd != tp.float64 else "floor"
                fn = wrap(lambda es: f"{base}({es[0]} / {es[1]})")
            return self._emit_binop(node, a, b, rd, cd, fn)
        if op == ast.Mod:
            if rd.is_int:
                fn = lambda es: f"({es[0]} % {es[1]})"
            else:
                base = "fmodf" if cd != tp.float64 else "fmod"
                fn = wrap(lambda es: f"{base}({es[0]}, {es[1]})")
            return self._emit_binop(node, a, b, rd, cd, fn)
        if op == ast.Pow:
            if b.layout == CONSTEXPR and b.pyval == 2:
                fn = wrap(lambda es: f"({es[0]} * {es[0]})")
                return self._emit_binop(node, a, a, rd, cd, fn)
            if rd.is_int:
                self.err(node, "integer ** requires a constexpr exponent of 2")
            base = "powf" if cd != tp.float64 else "pow"
            fn = wrap(lambda es: f"{base}({es[0]}, {es[1]})")
            return self._emit_binop(node, a, b, rd, cd, fn)
        self.err(node, f"unsupported operator {op.__name__}")

    def _emit_binop(self, node, a, b, rd, cd, fn):
        return self.elementwise(
            node, [a, b], rd,
            lambda es: fn([self._to_compute(e, o, cd, rd) for e, o in zip(es, (a, b))]))

    def _to_compute(self, expr, operand, cd, vd=None):
        if operand.layout == CONSTEXPR:
            # constants are rounded to the *value* dtype first (Triton rule:
            # fp16_block * 0.1 uses fp16(0.1), not fp32(0.1))
            vd = vd or cd
            return self.convert(self.literal(operand.pyval, vd), vd, cd)
        src = operand.dtype if not isinstance(operand.dtype, tp.pointer_type) else None
        if src is None:
            return expr
        return self.convert(expr, src, cd)

    def ptr_arith(self, node, op, a, b):
        if isinstance(b.dtype, tp.pointer_type) and not isinstance(a.dtype, tp.pointer_type):
            a, b = b, a
            if op == ast.Sub:
                self.err(node, "int - pointer is not supported")
        if op not in (ast.Add, ast.Sub):
            self.err(node, "only + and - on pointers")
        if isinstance(b.dtype, tp.pointer_type):
            self.err(node, "pointer - pointer not supported")
        neg = op == ast.Sub
        a = self.materialize(a) if a.layout == LAZY_ZERO else a
        # offsets int block; base C expr
        if a.layout == UNIFORM or a.layout == CONSTEXPR:
            base = a.var if a.layout == UNIFORM else None
            if base is None:
                self.err(node, "invalid pointer constexpr")
            if b.layout in (UNIFORM, CONSTEXPR):
                bexpr = b.var if b.layout == UNIFORM else self.literal(b.pyval, tp.int64)
                t = self.fresh("p")
                sign = "-" if neg else "+"
                self.emit(f"{a.dtype.ctype} {t} = {base} {sign} ({bexpr});")
                return Value(UNIFORM, a.dtype, (), t)
            # scalar ptr + int block -> pointer block with these offsets
            b = self.materialize(b)
            if not b.dtype.is_int:
                self.err(node, "pointer offsets must be integers")
            offs = b
            if neg:
                offs = self.elementwise(node, [b], b.dtype, lambda es: f"(-{es[0]})")
            return Value(CYCLIC, a.dtype, offs.shape, offs.var, base=base)
        # pointer block +/- int
        shape = tp.broadcast_shapes(a.shape, b.shape if b.layout == CYCLIC else ())
        vals, shape = self.broadcast_values(node, [a, b])
        a2, b2 = vals
        out = self.fresh("po")
        S = self.slots(math.prod(shape) if shape else 1)
        self.emit(f"int {out}[{S}];")
        g = self.slot_loop(math.prod(shape))
        ae = f"{a2.var}[_s]" if a2.layout == CYCLIC else a2.var
        if b2.layout == CONSTEXPR:
            be = self.literal(b2.pyval, tp.int32)
        elif b2.layout == UNIFORM:
            be = b2.var
        else:
            be = f"{b2.var}[_s]"
        sign = "-" if neg else "+"
        line = f"{out}[_s] = {ae} {sign} {be};"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, a.dtype, shape, out, base=a2.base)

    def expr_Compare(self, node):
        if len(node.ops) != 1:
            self.err(node, "chained comparisons not supported")
        a = self.visit_expr(node.left)
        b = self.visit_expr(node.comparators[0])
        if a.layout == CONSTEXPR and b.layout == CONSTEXPR:
            import operator

            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            return cx(ops[type(node.ops[0])](a.pyval, b.pyval))
        sym = _CMPOPS.get(type(node.ops[0]))
        if sym is None:
            self.err(node, f"unsupported comparison {type(node.ops[0]).__name__}")
        da, db = self.cdtype(a), self.cdtype(b)
        cd = tp.promote(da, db)
        if cd in (tp.float16, tp.bfloat16):
            cd = tp.float32
        return self.elementwise(
            node, [a, b], tp.int1,
            lambda es: f"({self._to_compute(es[0], a, cd)} {sym} {self._to_compute(es[1], b, cd)})")

    def expr_BoolOp(self, node):
        vals = [self.visit_expr(v) for v in node.values]
        if all(v.layout == CONSTEXPR for v in vals):
            import functools

            if isinstance(node.op, ast.And):
                return cx(functools.reduce(lambda x, y: x and y, [v.pyval for v in vals]))
            return cx(functools.reduce(lambda x, y: x or y, [v.pyval for v in vals]))
        if any(v.layout == CYCLIC for v in vals):
            self.err(node, "use & / | for elementwise boolean ops on blocks")
        sym = "&&" if isinstance(node.op, ast.And) else "||"
        exprs = []
        for v in vals:
            exprs.append(self.literal(v.pyval, tp.int1) if v.layout == CONSTEXPR else v.var)
        t = self.fresh()
        self.emit(f"bool {t} = ({f' {sym} '.join(exprs)});")
        return Value(UNIFORM, tp.int1, (), t)

    def expr_IfExp(self, node):
        c = self.visit_expr(node.test)
        if c.layout == CONSTEXPR:
            return self.visit_expr(node.body if c.pyval else node.orelse)
        a = self.visit_expr(node.body)
        b = self.visit_expr(node.orelse)
        return self.op_where(node, c, a, b)

    def expr_Subscript(self, node):
        base = self.visit_expr(node.value)
        if base.layout == CONSTEXPR:
            idx = self.visit_expr(node.slice)
            if idx.layout != CONSTEXPR:
                self.err(node, "runtime indexing of constexpr values")
            return cx(base.pyval[idx.pyval])
        # block indexing: only None / full-slice patterns (expand_dims)
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        newshape = []
        di = 0
        for e in elts:
            if isinstance(e, ast.Constant) and e.value is None:
                newshape.append(1)
            elif isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None:
                if di >= len(base.shape):
                    self.err(node, "too many indices")
                newshape.append(base.shape[di])
                di += 1
            else:
                self.err(node, "only x[:, None]-style indexing is supported on blocks")
        newshape += list(base.shape[di:])
        base = self.materialize(base)
        return Value(base.layout, base.dtype, tuple(newshape), base.var, base=base.base)

    def _callee_name(self, func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def expr_Call(self, node):
        # .to(dtype) works on any device value expression
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            base = self.visit_expr(node.func.value)
            if isinstance(base, Value) and base.layout != CONSTEXPR:
                dt = self.visit_expr(node.args[0])
                return self.cast(node, base, dt.pyval)
        name = self._callee_name(node.func)
        # try resolving to a python object to detect our builtins
        target = self._try_resolve(node.func)
        if target is not None and hasattr(target, "_newt_builtin"):
            name = target._newt_builtin
        args = [self.visit_expr(a) for a in node.args]
        kwargs = {kw.arg: self.visit_expr(kw.value) for kw in node.keywords}
        handler = getattr(self, f"op_{name}", None)
        if handler is not None:
            return handler(node, *args, **kwargs)
        # host-side helpers on constexprs (min/max/len/int/float/...)
        if target is not None and all(v.layout == CONSTEXPR for v in args) and all(
            v.layout == CONSTEXPR for v in kwargs.values()
        ):
            try:
                return cx(target(*[v.pyval for v in args],
                                 **{k: v.pyval for k, v in kwargs.items()}))
            except Exception as e:
                self.err(node, f"constexpr call failed: {e}")
        if name in ("min", "max", "abs"):
            if name == "abs":
                return self.op_abs(node, *args)
            return (self.op_minimum if name == "min" else self.op_maximum)(node, *args)
        self.err(node, f"unknown function '{name}' (newt kernels can only call nl.* builtins)")

    def _try_resolve(self, func):
        try:
            if isinstance(func, ast.Name):
                if func.id in self.locals and self.locals[func.id].layout == CONSTEXPR:
                    return self.locals[func.id].pyval
                if func.id in self.globals:
                    return self.globals[func.id]
                import builtins

                return getattr(builtins, func.id, None)
            if isinstance(func, ast.Attribute):
                base = self._try_resolve(func.value)
                if base is not None:
                    return getattr(base, func.attr, None)
        except Exception:
            return None
        return None

    # -- builtin ops -----------------------------------------------------------

    def _cexpect(self, node, v, what):
        if v.layout != CONSTEXPR:
            self.err(node, f"{what} must be a compile-time constant")
        return v.pyval

    def op_program_id(self, node, axis):
        ax = self._cexpect(node, axis, "program_id axis")
        t = self.fresh("pid")
        self.emit(f"int {t} = (int)blockIdx.{'xyz'[ax]};")
        return Value(UNIFORM, tp.int32, (), t)

    def op_num_programs(self, node, axis):
        ax = self._cexpect(node, axis, "num_programs axis")
        t = self.fresh("np")
        self.emit(f"int {t} = (int)gridDim.{'xyz'[ax]};")
        return Value(UNIFORM, tp.int32, (), t)

    def op_arange(self, node, start, end):
        s = self._cexpect(node, start, "arange start")
        e = self._cexpect(node, end, "arange end")
        n = e - s
        if n <= 0 or (n & (n - 1)) != 0:
            self.err(node, f"arange size must be a positive power of two, got {n}")
        out = self.fresh("ar")
        S = self.slots(n)
        self.emit(f"int {out}[{S}];")
        g = self.slot_loop(n)
        line = f"{out}[_s] = {s} + {self.lin_expr()};"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, tp.int32, (n,), out)

    def _shape_arg(self, node, shape):
        sh = self._cexpect(node, shape, "block shape")
        if isinstance(sh, int):
            sh = (sh,)
        sh = tuple(int(x) for x in sh)
        for d in sh:
            if d <= 0 or (d & (d - 1)) != 0:
                self.err(node, f"block dims must be powers of two, got {sh}")
        return sh

    def op_zeros(self, node, shape, dtype=None):
        sh = self._shape_arg(node, shape)
        dt = dtype.pyval if dtype is not None else tp.float32
        ph = len(self.lines)
        self.emit("/* zeros placeholder */")
        var = self.fresh("z")
        # frag meta is filled in if this becomes a dot accumulator
        v = Value(LAZY_ZERO, dt, sh, var, meta={"ph": ph, "frag_meta": None})
        return v

    def op_full(self, node, shape, value, dtype=None):
        sh = self._shape_arg(node, shape)
        dt = dtype.pyval if dtype is not None else tp.float32
        out = self.fresh("fl")
        S = self.slots(math.prod(sh))
        self.emit(f"{dt.ctype} {out}[{S}];")
        self.slot_loop(math.prod(sh))
        if value.layout == CONSTEXPR:
            src = self.literal(value.pyval, dt)
        elif value.layout == UNIFORM:
            src = self.convert(value.var, value.dtype, dt)
        else:
            self.err(node, "full() value must be a scalar")
        self.emit(f"{out}[_s] = {src};")
        self.end_loop()
        return Value(CYCLIC, dt, sh, out)

    def op_load(self, node, ptr, mask=None, other=None):
        if not isinstance(ptr.dtype, tp.pointer_type):
            self.err(node, "load() expects a pointer block")
        elem = ptr.dtype.element
        if elem == tp.float16:
            self.uses.add("fp16")
        if elem == tp.bfloat16:
            self.uses.add("bf16")
        if mask is not None and mask.layout in (LAZY_LOAD, LAZY_ZERO):
            mask = self.materialize(mask)
        if other is not None and other.layout in (LAZY_LOAD, LAZY_ZERO):
            other = self.materialize(other)
        if ptr.layout == UNIFORM:
            if mask is not None and mask.layout == CYCLIC:
                self.err(node, "a scalar pointer cannot take a block mask "
                               "(broadcast the pointer with `ptr + offsets` first)")
            t = self.fresh("ld")
            if mask is not None:
                mexpr = mask.var if mask.layout == UNIFORM else self.literal(mask.pyval, tp.int1)
                oexpr = (self.literal(other.pyval if other else 0, elem) if other is None or
                         other.layout == CONSTEXPR else self.convert(other.var, other.dtype, elem))
                self.emit(f"{elem.ctype} {t} = ({mexpr}) ? *({ptr.var}) : {oexpr};")
            else:
                self.emit(f"{elem.ctype} {t} = *({ptr.var});")
            return Value(UNIFORM, elem, (), t)
        operands = [ptr] + ([mask] if mask is not None else []) + (
            [other] if other is not None and other.layout == CYCLIC else [])
        operands, shape = self.broadcast_values(node, operands)
        ptr = operands[0]
        mask = operands[1] if mask is not None else None
        if other is not None and other.layout == CYCLIC:
            other = operands[-1]
        # defer emission: a dot can stage this straight to smem with cp.async;
        # any other use materializes it (flush points guard memory mutation)
        lazy = Value(LAZY_LOAD, elem, shape, None,
                     meta={"ptr": ptr, "mask": mask, "other": other})
        if not hasattr(self, "pending_lazy"):
            self.pending_lazy = []
        self.pending_lazy.append(lazy)
        return lazy

    def _materialize_load(self, v):
        """Emit the deferred load; mutates v into a CYCLIC register block."""
        if v.meta.get("poisoned"):
            raise CompileError(
                "a value from nl.load was consumed by nl.dot and then used after "
                "a store/loop boundary; load it into a separate variable first")
        ptr, mask, other = v.meta["ptr"], v.meta["mask"], v.meta["other"]
        elem, shape = v.dtype, v.shape
        out = self.fresh("ld")
        numel = math.prod(shape)
        S = self.slots(numel)
        self.emit(f"{elem.ctype} {out}[{S}];")

        def scalar_body(conds):
            if other is None:
                oexpr = self.literal(0, elem)
            elif other.layout == CONSTEXPR:
                oexpr = self.literal(other.pyval, elem)
            elif other.layout == UNIFORM:
                oexpr = self.convert(other.var, other.dtype, elem)
            else:
                oexpr = self.convert(f"{other.var}[_s]", other.dtype, elem)
            gexpr = f"{ptr.base}[{ptr.var}[_s]]"
            if conds:
                self.emit(f"{out}[_s] = ({' && '.join(conds)}) ? {gexpr} : {oexpr};")
            else:
                self.emit(f"{out}[_s] = {gexpr};")

        def mask_lane(s_expr):
            if mask is None:
                return None
            if mask.layout == CYCLIC:
                return f"{mask.var}[{s_expr}]"
            if mask.layout == UNIFORM:
                return mask.var
            return self.literal(mask.pyval, tp.int1)

        V = self.VEC
        gbytes = V * elem.itemsize
        if V > 1 and gbytes in (4, 8, 16, 32) and S % V == 0:
            self._emit_vector_access(
                "load", numel, S, V, elem, ptr, mask_lane, scalar_body, out=out, other=other)
        else:
            g = self.slot_loop(numel)
            conds = [c for c in [g, mask_lane("_s")] if c]
            scalar_body(conds)
            self.end_loop()
        v.layout = CYCLIC
        v.var = out
        v.meta = None
        return v

    def _emit_vector_access(self, kind, numel, S, V, elem, ptr, mask_lane, scalar_body,
                            out=None, other=None, value_expr=None):
        """Group loop with a runtime-checked vector fast path + scalar fallback."""
        self.uses.add("vec")
        align = min(V * elem.itemsize, 16)
        wrapper = f"_nv<{elem.ctype}, {V}, {align}>"
        guard = self.guard(numel)
        self.emit("#pragma unroll")
        self.emit(f"for (int _g = 0; _g < {S // V}; ++_g) {{")
        self.indent += 1
        self.emit(f"int _i0 = _g * {V};")
        self.emit(f"int _o0 = {ptr.var}[_i0];")
        conds = []
        # group entirely within the block?
        if guard is not None:
            conds.append(f"((_g * {self.T} + _tid) * {V} + {V}) <= {numel}")
        # contiguous offsets
        contig = " && ".join(f"{ptr.var}[_i0 + {v}] == _o0 + {v}" for v in range(1, V))
        conds.append(f"({contig})")
        # aligned base
        conds.append(f"((((unsigned long long)(&{ptr.base}[_o0])) & {align - 1}) == 0)")
        # mask uniform-true across the group
        m0 = mask_lane("_i0")
        if m0 is not None:
            allm = " && ".join(str(mask_lane(f"_i0 + {v}")) for v in range(V))
            conds.append(f"({allm})")
        self.emit(f"if ({' && '.join(conds)}) {{")
        self.indent += 1
        if kind == "load":
            self.emit(f"{wrapper} _w = *(const {wrapper}*)(&{ptr.base}[_o0]);")
            self.emit("#pragma unroll")
            self.emit(f"for (int _v = 0; _v < {V}; ++_v) {out}[_i0 + _v] = _w.d[_v];")
        else:
            self.emit(f"{wrapper} _w;")
            self.emit("#pragma unroll")
            self.emit(f"for (int _v = 0; _v < {V}; ++_v) {{ int _s = _i0 + _v; "
                      f"_w.d[_v] = {value_expr('_s')}; }}")
            self.emit(f"*({wrapper}*)(&{ptr.base}[_o0]) = _w;")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit("#pragma unroll")
        self.emit(f"for (int _v = 0; _v < {V}; ++_v) {{")
        self.indent += 1
        self.emit("int _s = _i0 + _v;")
        conds = [c for c in [guard, mask_lane("_s")] if c]
        scalar_body(conds)
        self.end_loop()
        self.end_loop()
        self.end_loop()

    def op_store(self, node, ptr, value, mask=None):
        if not isinstance(ptr.dtype, tp.pointer_type):
            self.err(node, "store() expects a pointer block")
        elem = ptr.dtype.element
        value = self.materialize(value, CYCLIC)
        if mask is not None and mask.layout in (LAZY_LOAD, LAZY_ZERO):
            mask = self.materialize(mask)
        self.flush_lazy_loads()  # deferred loads must read pre-store memory
        self._drain_pipelines()  # in-flight cp.async sources must stay immutable
        if ptr.layout == UNIFORM:
            if mask is not None and mask.layout == CYCLIC:
                self.err(node, "a scalar pointer cannot take a block mask")
            if value.layout == CYCLIC:
                self.err(node, "cannot store a block through a scalar pointer")
            src = (self.literal(value.pyval, elem) if value.layout == CONSTEXPR
                   else self.convert(value.var, value.dtype, elem))
            if mask is not None and mask.layout == CONSTEXPR:
                if not mask.pyval:
                    return cx(None)
                mask = None
            if mask is not None and mask.layout == UNIFORM:
                self.emit(f"if ({mask.var}) *({ptr.var}) = {src};")
            else:
                self.emit(f"*({ptr.var}) = {src};")
            return cx(None)
        operands = [ptr] + ([value] if value.layout == CYCLIC else []) + (
            [mask] if mask is not None and mask.layout == CYCLIC else [])
        operands, shape = self.broadcast_values(node, operands)
        ptr = operands[0]
        if value.layout == CYCLIC:
            value = operands[1]
        if mask is not None and mask.layout == CYCLIC:
            mask = operands[-1]
        numel = math.prod(shape)
        S = self.slots(numel)

        def value_expr(s_expr):
            if value.layout == CONSTEXPR:
                return self.literal(value.pyval, elem)
            if value.layout == UNIFORM:
                return self.convert(value.var, value.dtype, elem)
            return self.convert(f"{value.var}[{s_expr}]", value.dtype, elem)

        def mask_lane(s_expr):
            if mask is None:
                return None
            if mask.layout == CYCLIC:
                return f"{mask.var}[{s_expr}]"
            if mask.layout == UNIFORM:
                return mask.var
            return "false" if not mask.pyval else None

        def scalar_body(conds):
            line = f"{ptr.base}[{ptr.var}[_s]] = {value_expr('_s')};"
            self.emit(f"if ({' && '.join(conds)}) {line}" if conds else line)

        V = self.VEC
        gbytes = V * elem.itemsize
        if V > 1 and gbytes in (4, 8, 16, 32) and S % V == 0:
            self._emit_vector_access(
                "store", numel, S, V, elem, ptr, mask_lane, scalar_body, value_expr=value_expr)
        else:
            g = self.slot_loop(numel)
            conds = [c for c in [g, mask_lane("_s")] if c]
            scalar_body(conds)
            self.end_loop()
        return cx(None)

    def _atomic(self, node, fn, ptr, value, mask):
        if not isinstance(ptr.dtype, tp.pointer_type):
            self.err(node, "atomic ops expect a pointer block")
        elem = ptr.dtype.element
        value = self.materialize(value, CYCLIC)
        self.flush_lazy_loads()
        self._drain_pipelines()
        operands = [ptr] + ([value] if value.layout == CYCLIC else []) + (
            [mask] if mask is not None and mask.layout == CYCLIC else [])
        operands, shape = self.broadcast_values(node, operands)
        ptr = operands[0]
        if value.layout == CYCLIC:
            value = operands[1]
        if mask is not None and mask.layout == CYCLIC:
            mask = operands[-1]
        numel = math.prod(shape) if shape else 1
        if ptr.layout == UNIFORM:
            if value.layout == CYCLIC:
                self.err(node, "scalar-pointer atomic with a block value; reduce it first")
            if mask is not None and mask.layout == CYCLIC:
                self.err(node, "scalar-pointer atomic cannot take a block mask")
            src = (self.literal(value.pyval, elem) if value.layout == CONSTEXPR
                   else self.convert(value.var, value.dtype, elem))
            # one atomic per *program*, not per thread
            conds = ["_tid == 0"]
            if mask is not None and mask.layout == UNIFORM:
                conds.append(mask.var)
            self.emit(f"if ({' && '.join(conds)}) {fn}({ptr.var}, {src});")
            return cx(None)
        g = self.slot_loop(numel)
        conds = [c for c in [g] if c]
        if mask is not None:
            conds.append(f"{mask.var}[_s]" if mask.layout == CYCLIC else mask.var)
        if value.layout == CONSTEXPR:
            src = self.literal(value.pyval, elem)
        elif value.layout == UNIFORM:
            src = self.convert(value.var, value.dtype, elem)
        else:
            src = self.convert(f"{value.var}[_s]", value.dtype, elem)
        line = f"{fn}(&{ptr.base}[{ptr.var}[_s]], {src});"
        self.emit(f"if ({' && '.join(conds)}) {line}" if conds else line)
        self.end_loop()
        return cx(None)

    def op_atomic_add(self, node, ptr, value, mask=None):
        return self._atomic(node, "atomicAdd", ptr, value, mask)

    def op_atomic_max(self, node, ptr, value, mask=None):
        if isinstance(ptr.dtype, tp.pointer_type) and ptr.dtype.element == tp.float32:
            self.uses.add("atomicmaxf")  # no atomicMax(float*) overload: CAS loop
            return self._atomic(node, "_nv_atomic_max_f32", ptr, value, mask)
        return self._atomic(node, "atomicMax", ptr, value, mask)

    # shape ops
    def op_expand_dims(self, node, x, axis):
        ax = self._cexpect(node, axis, "axis")
        x = self.materialize(x)
        sh = list(x.shape)
        if ax < 0:
            ax += len(sh) + 1
        sh.insert(ax, 1)
        return Value(x.layout, x.dtype, tuple(sh), x.var, base=x.base)

    def op_broadcast_to(self, node, x, shape):
        sh = self._shape_arg(node, shape)
        return self.broadcast_to(node, x, sh)

    def op_reshape(self, node, x, shape):
        sh = self._shape_arg(node, shape)
        x = self.materialize(x)
        if math.prod(sh) != x.numel:
            self.err(node, f"reshape {x.shape} -> {sh} changes element count")
        return Value(x.layout, x.dtype, sh, x.var, base=x.base)

    def op_trans(self, node, x):
        x = self.materialize(x)
        if len(x.shape) != 2:
            self.err(node, "trans expects a 2D block")
        M, N = x.shape
        ct = "int" if x.is_ptr else x.dtype.ctype
        esize = 4 if x.is_ptr else x.dtype.itemsize
        pad = 1 if esize >= 4 else 2
        ld = N + pad
        self.track_smem(M * ld * esize)
        buf = self.fresh("tr")
        self.emit("__syncthreads();")
        self.emit(f"{ct}* {buf} = ({ct}*)_smem;")
        g = self.slot_loop(M * N)
        self.emit(f"int _j = {self.lin_expr()};")
        line = f"{buf}[(_j / {N}) * {ld} + (_j % {N})] = {x.var}[_s];"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        self.emit("__syncthreads();")
        out = self.fresh()
        S = self.slots(M * N)
        self.emit(f"{ct} {out}[{S}];")
        g = self.slot_loop(M * N)
        self.emit(f"int _j = {self.lin_expr()};")
        # output is [N, M]; element (n, m) = input (m, n)
        line = f"{out}[_s] = {buf}[(_j % {M}) * {ld} + (_j / {M})];"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, x.dtype, (N, M), out, base=x.base)

    def cast(self, node, x, dt):
        if not isinstance(dt, tp.dtype):
            self.err(node, f"invalid dtype {dt!r}")
        if dt == tp.float16:
            self.uses.add("fp16")
        if dt == tp.bfloat16:
            self.uses.add("bf16")
        x = self.materialize(x, CYCLIC)
        if x.layout == CONSTEXPR:
            return cx(x.pyval)
        if x.dtype == dt:
            return x
        if x.layout == UNIFORM:
            t = self.fresh()
            self.emit(f"{dt.ctype} {t} = {self.convert(x.var, x.dtype, dt)};")
            return Value(UNIFORM, dt, (), t)
        src = x.dtype
        return self.elementwise(node, [x], dt, lambda es: self.convert(es[0], src, dt))

    def op_cast(self, node, x, dtype):
        return self.cast(node, x, dtype.pyval)

    # reductions
    _RED = {
        "sum": ("+",),
        "max": ("max",),
        "min": ("min",),
    }

    def _red_identity(self, op, dt):
        if op == "sum":
            return self.literal(0, dt)
        if op == "max":
            if dt.is_floating:
                return self.literal(float("-inf"), dt)
            return "(-2147483647 - 1)" if dt == tp.int32 else "(-9223372036854775807LL - 1)"
        if op == "min":
            if dt.is_floating:
                return self.literal(float("inf"), dt)
            return "2147483647" if dt == tp.int32 else "9223372036854775807LL"

    def _red_combine(self, op, dt, a, b):
        if op == "sum":
            return f"({a} + {b})"
        if dt.is_floating:
            fn = ("fmaxf" if op == "max" else "fminf") if dt != tp.float64 else (
                "fmax" if op == "max" else "fmin")
            return f"{fn}({a}, {b})"
        return f"({op})({a}, {b})".replace("(max)", "max").replace("(min)", "min")

    def _reduce(self, node, x, axis, op):
        x = self.materialize(x, CYCLIC)
        if x.layout == UNIFORM or x.layout == CONSTEXPR:
            return x
        rt = tp.float32 if x.dtype in (tp.float16, tp.bfloat16) else x.dtype
        if rt == tp.int1:
            rt = tp.int32  # sum(mask) counts, max(mask) is 'any' - bool math can't
        if axis is not None:
            ax = self._cexpect(node, axis, "axis") if isinstance(axis, Value) else axis
            if ax < 0:
                ax += len(x.shape)
            if len(x.shape) == 1 and ax == 0:
                return self._reduce_full(node, x, op, rt)
            return self._reduce_axis(node, x, ax, op, rt)
        return self._reduce_full(node, x, op, rt)

    def _reduce_full(self, node, x, op, rt):
        p = self.fresh("rp")
        ident = self._red_identity(op, rt)
        self.emit(f"{rt.ctype} {p} = {ident};")
        g = self.slot_loop(x.numel)
        e = self.convert(f"{x.var}[_s]", x.dtype, rt)
        line = f"{p} = {self._red_combine(op, rt, p, e)};"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        # warp reduction
        self.emit("#pragma unroll")
        self.emit("for (int _o = 16; _o > 0; _o >>= 1) "
                  f"{p} = {self._red_combine(op, rt, p, f'__shfl_xor_sync(0xffffffffu, {p}, _o)')};")
        if self.num_warps > 1:
            self.track_smem(32 * rt.itemsize)
            buf = self.fresh("rb")
            self.emit("__syncthreads();")
            self.emit(f"{rt.ctype}* {buf} = ({rt.ctype}*)_smem;")
            self.emit(f"if (_lane == 0) {buf}[_warp] = {p};")
            self.emit("__syncthreads();")
            self.emit(f"{p} = (_tid < {self.num_warps}) ? {buf}[_tid] : {ident};")
            self.emit("#pragma unroll")
            self.emit("for (int _o = 16; _o > 0; _o >>= 1) "
                      f"{p} = {self._red_combine(op, rt, p, f'__shfl_xor_sync(0xffffffffu, {p}, _o)')};")
            # only warp 0 holds the result; broadcast through smem
            self.emit(f"if (_tid == 0) {buf}[0] = {p};")
            self.emit("__syncthreads();")
            self.emit(f"{p} = {buf}[0];")
        return Value(UNIFORM, rt, (), p)

    def _reduce_axis(self, node, x, ax, op, rt):
        shape = x.shape
        if ax >= len(shape):
            self.err(node, f"axis {ax} out of range for shape {shape}")
        R = shape[ax]
        oshape = tuple(d for i, d in enumerate(shape) if i != ax)
        onumel = math.prod(oshape) if oshape else 1
        if onumel == 1:
            v = self._reduce_full(node, x, op, rt)
            if oshape:
                # keep as a block of size-1 dims? result shape has dims; broadcast scalar
                return self.broadcast_to(node, Value(UNIFORM, rt, (), v.var), oshape)
            return v
        # stage x into smem, each thread reduces its output elements serially
        ct = x.dtype.ctype
        self.track_smem(x.numel * x.dtype.itemsize)
        buf = self.fresh("rs")
        self.emit("__syncthreads();")
        self.emit(f"{ct}* {buf} = ({ct}*)_smem;")
        g = self.slot_loop(x.numel)
        line = f"{buf}[{self.lin_expr()}] = {x.var}[_s];"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        self.emit("__syncthreads();")
        strides = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * shape[d + 1]
        ostrides = [1] * len(oshape)
        for d in range(len(oshape) - 2, -1, -1):
            ostrides[d] = ostrides[d + 1] * oshape[d + 1]
        out = self.fresh("rr")
        S = self.slots(onumel)
        self.emit(f"{rt.ctype} {out}[{S}];")
        g = self.slot_loop(onumel)
        self.emit(f"int _j = {self.lin_expr()};")
        terms = []
        srcdims = [i for i in range(len(shape)) if i != ax]
        for k, d in enumerate(srcdims):
            coord = f"((_j / {ostrides[k]}) % {oshape[k]})"
            terms.append(f"{coord} * {strides[d]}" if strides[d] != 1 else coord)
        base = " + ".join(terms) if terms else "0"
        self.emit(f"int _base = {base};")
        acc = self.fresh("ra")
        self.emit(f"{rt.ctype} {acc} = {self._red_identity(op, rt)};")
        body = f"{acc} = {self._red_combine(op, rt, acc, self.convert(f'{buf}[_base + _r * {strides[ax]}]', x.dtype, rt))};"
        self.emit(f"for (int _r = 0; _r < {R}; ++_r) {body}")
        line = f"{out}[_s] = {acc};"
        self.emit(f"if ({g}) {line}" if g else line)
        self.end_loop()
        return Value(CYCLIC, rt, oshape, out)

    def op_sum(self, node, x, axis=None):
        return self._reduce(node, x, axis, "sum")

    def op_max(self, node, x, axis=None):
        return self._reduce(node, x, axis, "max")

    def op_min(self, node, x, axis=None):
        return self._reduce(node, x, axis, "min")

    # math
    def _math1(self, node, x, name):
        x = self.materialize(x, CYCLIC)
        d = self.cdtype(x)
        if not d.is_floating:
            x = self.cast(node, x, tp.float32)
            d = tp.float32
        cd = tp.float32 if d in (tp.float16, tp.bfloat16) else d
        f32, f64 = _MATH_FNS[name]
        fn = f64 if cd == tp.float64 else f32
        src = d

        def expr(es):
            e = self.convert(es[0], src, cd) if src != cd else es[0]
            if name == "rsqrt":
                r = f"rsqrtf({e})" if cd != tp.float64 else f"rsqrt({e})"
            else:
                r = f"{fn}({e})"
            return self.convert(r, cd, d) if cd != d else r

        return self.elementwise(node, [x], d, expr)

    def op_exp(self, node, x):
        return self._math1(node, x, "exp")

    def op_exp2(self, node, x):
        return self._math1(node, x, "exp2")

    def op_log(self, node, x):
        return self._math1(node, x, "log")

    def op_log2(self, node, x):
        return self._math1(node, x, "log2")

    def op_sqrt(self, node, x):
        return self._math1(node, x, "sqrt")

    def op_rsqrt(self, node, x):
        return self._math1(node, x, "rsqrt")

    def op_sin(self, node, x):
        return self._math1(node, x, "sin")

    def op_cos(self, node, x):
        return self._math1(node, x, "cos")

    def op_tanh(self, node, x):
        return self._math1(node, x, "tanh")

    def op_erf(self, node, x):
        return self._math1(node, x, "erf")

    def op_floor(self, node, x):
        return self._math1(node, x, "floor")

    def op_ceil(self, node, x):
        return self._math1(node, x, "ceil")

    def op_sigmoid(self, node, x):
        x = self.materialize(x, CYCLIC)
        d = self.cdtype(x)
        cd = tp.float32 if d in (tp.float16, tp.bfloat16) else d
        src = d

        def expr(es):
            e = self.convert(es[0], src, cd) if src != cd else es[0]
            one = "1.0" if cd == tp.float64 else "1.0f"
            ex = "exp" if cd == tp.float64 else "expf"
            r = f"({one} / ({one} + {ex}(-({e}))))"
            return self.convert(r, cd, d) if cd != d else r

        return self.elementwise(node, [x], d, expr)

    def op_abs(self, node, x):
        if x.layout == CONSTEXPR:
            return cx(abs(x.pyval))
        x = self.materialize(x, CYCLIC)
        d = self.cdtype(x)
        if d.is_floating:
            cd = tp.float32 if d in (tp.float16, tp.bfloat16) else d
            fn = "fabs" if cd == tp.float64 else "fabsf"
            src = d
            return self.elementwise(
                node, [x], d,
                lambda es: self.convert(f"{fn}({self.convert(es[0], src, cd)})", cd, d)
                if cd != d else f"{fn}({es[0]})")
        return self.elementwise(node, [x], d, lambda es: f"abs({es[0]})")

    def _math2(self, node, a, b, ffn, ifn=None):
        da, db = self.cdtype(a), self.cdtype(b)
        rd = tp.promote(da, db)
        cd = tp.float32 if rd in (tp.float16, tp.bfloat16) else rd
        if rd.is_int and ifn is None:
            self.err(node, "op requires floating point")
        fn = ifn if rd.is_int else (ffn if cd != tp.float64 else ffn.rstrip("f"))

        def expr(es):
            e0 = self._to_compute(es[0], a, cd)
            e1 = self._to_compute(es[1], b, cd)
            r = f"{fn}({e0}, {e1})"
            return self.convert(r, cd, rd) if cd != rd else r

        return self.elementwise(node, [a, b], rd, expr)

    def op_maximum(self, node, a, b):
        return self._math2(node, a, b, "fmaxf", "max")

    def op_minimum(self, node, a, b):
        return self._math2(node, a, b, "fminf", "min")

    def op_fma(self, node, a, b, c):
        da = tp.promote(self.cdtype(a), tp.promote(self.cdtype(b), self.cdtype(c)))
        cd = tp.float32 if da in (tp.float16, tp.bfloat16) else da
        fn = "fma" if cd == tp.float64 else "fmaf"

        def expr(es):
            parts = [self._to_compute(e, o, cd) for e, o in zip(es, (a, b, c))]
            r = f"{fn}({parts[0]}, {parts[1]}, {parts[2]})"
            return self.convert(r, cd, da) if cd != da else r

        return self.elementwise(node, [a, b, c], da, expr)

    def op_where(self, node, cond, a, b):
        if cond.layout == CONSTEXPR:
            return a if cond.pyval else b
        da, db = self.cdtype(a), self.cdtype(b)
        rd = tp.promote(da, db)

        def expr(es):
            c = es[0]
            x = self._to_compute(es[1], a, rd)
            y = self._to_compute(es[2], b, rd)
            return f"(({c}) ? ({x}) : ({y}))"

        return self.elementwise(node, [cond, a, b], rd, expr)

    def op_cdiv(self, node, a, b):
        if a.layout == CONSTEXPR and b.layout == CONSTEXPR:
            return cx(-(-a.pyval // b.pyval))
        rd = tp.promote(self.cdtype(a), self.cdtype(b))
        return self.elementwise(
            node, [a, b], rd,
            lambda es: f"(({self._to_compute(es[0], a, rd)} + {self._to_compute(es[1], b, rd)} - 1)"
                       f" / {self._to_compute(es[1], b, rd)})")

    def op_multiple_of(self, node, x, value=None):
        return x

    def op_max_contiguous(self, node, x, value=None):
        return x

    def op_static_assert(self, node, cond, msg=None):
        c = self._cexpect(node, cond, "static_assert condition")
        if not c:
            m = msg.pyval if msg is not None else ""
            self.err(node, f"static_assert failed: {m}")
        return cx(None)

    def op_static_print(self, node, *args):
        print("[newt static_print]", *[a.pyval if a.layout == CONSTEXPR else a for a in args])
        return cx(None)

    # -- dot -------------------------------------------------------------------

    def op_dot(self, node, a, b, acc=None, out_dtype=None):
        use_async = (a.layout == LAZY_LOAD and b.layout == LAZY_LOAD
                     and not a.meta.get("poisoned") and not b.meta.get("poisoned")
                     and os.environ.get("NEWT_ASYNC_DOT", "1") != "0")
        if not use_async:
            a = self.materialize(a, CYCLIC)
            b = self.materialize(b, CYCLIC)
        if len(a.shape) != 2 or len(b.shape) != 2:
            self.err(node, "dot expects 2D blocks")
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            self.err(node, f"dot shape mismatch: {a.shape} @ {b.shape}")
        if a.dtype != b.dtype:
            self.err(node, f"dot operands must share a dtype ({a.dtype} vs {b.dtype})")
        if a.dtype == tp.float16:
            ft, KF = "__half", 16
            self.uses.add("fp16")
        elif a.dtype == tp.bfloat16:
            ft, KF = "__nv_bfloat16", 16
            self.uses.add("bf16")
        elif a.dtype == tp.float32:
            ft, KF = "wmma::precision::tf32", 8
        else:
            self.err(node, f"dot supports fp16/bf16/fp32 inputs, got {a.dtype}")
        if M < 16 or N < 16 or K < KF or M % 16 or N % 16 or K % KF:
            self.err(node, f"dot requires M,N multiples of 16 and K multiple of {KF} "
                           f"(got {M}x{K} @ {K}x{N})")
        self.uses.add("mma")
        m = self._frag_meta(M, N, KF)
        # a persistent (loop-carried) fragment accumulator can be pipelined:
        # each execution stages its tile async and runs the mma for the tile
        # staged by the PREVIOUS execution, hiding a full iteration of memory
        # latency. A CYCLIC acc (rescaled between dots, e.g. attention) can't:
        # its value is consumed every iteration.
        acc_streamable = acc is not None and acc.layout in (LAZY_ZERO, FRAG)
        # accumulator fragments
        if acc is None:
            accv = self.fresh("acc")
            self.emit(self._frag_decl(accv, m))
            self.emit("#pragma unroll")
            self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
            self.emit("#pragma unroll")
            self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn) "
                      f"wmma::fill_fragment({accv}[_fm][_fn], 0.0f);")
            acc = Value(FRAG, tp.float32, (M, N), accv, meta=m)
        else:
            if acc.layout == LAZY_ZERO:
                acc.meta["frag_meta"] = m
                self._materialize_lazy(acc, FRAG, None)
                acc.shape = (M, N)
            elif acc.layout == CYCLIC:
                # accumulator was used elementwise between dots (e.g. online
                # softmax rescaling) - pull it back into fragments
                if acc.shape != (M, N) or acc.dtype != tp.float32:
                    self.err(node, f"dot accumulator must be a float32 [{M},{N}] block, "
                                   f"got {acc.dtype} {acc.shape}")
                acc = self.cyclic_to_frag(acc, m)
            elif acc.meta != m:
                self.err(node, "dot accumulator tiling mismatch")
        # a deferred mma from another dot site must land before we touch acc
        if acc.var in self.frag_pending:
            self._emit_flush(acc.var)
        # staging geometry
        elem = a.dtype
        ces = elem.ctype
        esz = elem.itemsize
        PA = 8 if esz <= 2 else 4
        PB = 8 if esz <= 2 else 4
        lda, ldb = K + PA, N + PB
        bytesA = (M * lda * esz + 15) // 16 * 16
        bytesB = (K * ldb * esz + 15) // 16 * 16
        V = self.VEC
        CK = max(KF, V)
        async_ok = (V > 1 and V * esz in (4, 8, 16, 32)
                    and K % V == 0 and N % V == 0 and K % CK == 0)
        if use_async and not async_ok:
            use_async = False
            a = self.materialize(a, CYCLIC)
            b = self.materialize(b, CYCLIC)
        pipelined = (use_async and acc_streamable
                     and os.environ.get("NEWT_PIPELINE_DOT", "1") != "0")
        if pipelined:
            # the ring must fit next to the scratch arena (estimate the
            # epilogue's banded conversion buffer); otherwise fall back to
            # the chunked path rather than failing at assemble time
            est_scratch = max(self.smem_bytes, 16 * (N + 8) * 4)
            est_scratch = (est_scratch + 15) // 16 * 16
            rings = sum(self.ring_reservations) + 2 * (bytesA + bytesB)
            if est_scratch + rings > MAX_SMEM:
                pipelined = False

        if pipelined:
            # cross-iteration double buffering with deferred consumption:
            #   sync; cp.async THIS tile -> ring[buf^1]; commit;
            #   if pending: wait for the PREVIOUS tile; sync; mma(ring[buf]);
            #   rotate. The last staged tile is consumed by a flush at the
            #   first downstream read of the accumulator.
            self.uses.add("pipeline")
            a.meta["consumed"] = True
            b.meta["consumed"] = True
            site = len(self.ring_reservations)
            slotsz = bytesA + bytesB
            self.ring_reservations.append(2 * slotsz)
            buf = f"_dpb{site}"
            pend = f"_dpp{site}"
            self._prologue_decls.append(f"int {buf} = 0; bool {pend} = false;")
            self.emit("__syncthreads();")
            self.emit("{")
            self.indent += 1
            self.emit(f"{ces}* _Aw = ({ces}*)(_smem + _NRB{site} + ({buf} ^ 1) * {slotsz});")
            self.emit(f"{ces}* _Bw = ({ces}*)(_smem + _NRB{site} + ({buf} ^ 1) * {slotsz} "
                      f"+ {bytesA});")
            self._stage_chunk_async(a, M, K, lda, "_Aw", None, CK, by_rows=False)
            self._stage_chunk_async(b, K, N, ldb, "_Bw", None, CK, by_rows=True)
            self.emit("__pipeline_commit();")
            self.emit(f"if ({pend}) {{")
            self.indent += 1
            self.emit("__pipeline_wait_prior(1);")
            self.emit("__syncthreads();")
            self.emit(f"{ces}* _Ar = ({ces}*)(_smem + _NRB{site} + {buf} * {slotsz});")
            self.emit(f"{ces}* _Br = ({ces}*)(_smem + _NRB{site} + {buf} * {slotsz} "
                      f"+ {bytesA});")
            self._emit_mma(m, acc.var, "_Ar", "_Br", lda, ldb, ft, KF, 0, K // KF)
            self.indent -= 1
            self.emit("}")
            self.emit(f"{buf} ^= 1; {pend} = true;")
            self.indent -= 1
            self.emit("}")
            self.frag_pending.setdefault(acc.var, []).append(dict(
                site=site, m=m, lda=lda, ldb=ldb, ft=ft, KF=KF, K=K,
                ces=ces, slotsz=slotsz, bytesA=bytesA))
            return acc

        self.track_smem(bytesA + bytesB)
        As = self.fresh("As")
        Bs = self.fresh("Bs")
        self.emit("__syncthreads();")
        self.emit(f"{ces}* {As} = ({ces}*)_smem;")
        self.emit(f"{ces}* {Bs} = ({ces}*)(_smem + {bytesA});")
        if use_async:
            # chunked cp.async pipeline: copy chunk ck+1 (global -> smem via
            # the DMA path, no registers) while tensor cores chew on chunk ck
            self.uses.add("pipeline")
            CH = K // CK
            a.meta["consumed"] = True
            b.meta["consumed"] = True
            self._stage_chunk_async(a, M, K, lda, As, 0, CK, by_rows=False)
            self._stage_chunk_async(b, K, N, ldb, Bs, 0, CK, by_rows=True)
            self.emit("__pipeline_commit();")
            for ck in range(CH):
                if ck + 1 < CH:
                    self._stage_chunk_async(a, M, K, lda, As, ck + 1, CK, by_rows=False)
                    self._stage_chunk_async(b, K, N, ldb, Bs, ck + 1, CK, by_rows=True)
                    self.emit("__pipeline_commit();")
                self.emit(f"__pipeline_wait_prior({1 if ck + 1 < CH else 0});")
                self.emit("__syncthreads();")
                self._emit_mma(m, acc.var, As, Bs, lda, ldb, ft, KF,
                               ck * CK // KF, (ck + 1) * CK // KF)
        else:
            g = self.slot_loop(M * K)
            self.emit(f"int _j = {self.lin_expr()};")
            line = f"{As}[(_j / {K}) * {lda} + (_j % {K})] = {a.var}[_s];"
            self.emit(f"if ({g}) {line}" if g else line)
            self.end_loop()
            g = self.slot_loop(K * N)
            self.emit(f"int _j = {self.lin_expr()};")
            line = f"{Bs}[(_j / {N}) * {ldb} + (_j % {N})] = {b.var}[_s];"
            self.emit(f"if ({g}) {line}" if g else line)
            self.end_loop()
            self.emit("__syncthreads();")
            self._emit_mma(m, acc.var, As, Bs, lda, ldb, ft, KF, 0, K // KF)
        return acc

    def _emit_flush(self, accvar):
        """Run the deferred mma(s) of pipelined dot sites for this
        accumulator (runtime no-op if none pending). Emitted at every
        downstream read; at most one site's flag is true at runtime and the
        flags make repeated flush blocks idempotent."""
        for p in self.frag_pending[accvar]:
            site, ces, slotsz = p["site"], p["ces"], p["slotsz"]
            buf, pend = f"_dpb{site}", f"_dpp{site}"
            self.emit(f"if ({pend}) {{")
            self.indent += 1
            self.emit("__pipeline_wait_prior(0);")
            self.emit("__syncthreads();")
            self.emit(f"{ces}* _Ar = ({ces}*)(_smem + _NRB{site} + {buf} * {slotsz});")
            self.emit(f"{ces}* _Br = ({ces}*)(_smem + _NRB{site} + {buf} * {slotsz} "
                      f"+ {p['bytesA']});")
            self._emit_mma(p["m"], accvar, "_Ar", "_Br", p["lda"], p["ldb"],
                           p["ft"], p["KF"], 0, p["K"] // p["KF"])
            self.emit(f"{pend} = false;")
            self.indent -= 1
            self.emit("}")

    def _drain_pipelines(self):
        """Retire every pending deferred mma. Called before stores/atomics:
        a store could overwrite memory a staged cp.async is still reading
        (the async-copy source must stay immutable until the copy lands)."""
        for accvar in list(self.frag_pending):
            self._emit_flush(accvar)

    def _emit_mma(self, m, accv, As, Bs, lda, ldb, ft, KF, kk0, kk1):
        """Warp-level mma over fragment steps [kk0, kk1). Preloads the
        A-column / B-row fragments, then the FM x FN outer-product nest.
        Warps >= W idle (small dots); no syncs inside this block."""
        self.emit(f"if (_warp < {m['W']}) {{")
        self.indent += 1
        self.emit(f"int _wm = _warp / {m['WN']}, _wn = _warp % {m['WN']};")
        self.emit("#pragma unroll")
        self.emit(f"for (int _kk = {kk0}; _kk < {kk1}; ++_kk) {{")
        self.indent += 1
        self.emit(f"wmma::fragment<wmma::matrix_a, 16, 16, {KF}, {ft}, wmma::row_major> "
                  f"_af[{m['FM']}];")
        self.emit(f"wmma::fragment<wmma::matrix_b, 16, 16, {KF}, {ft}, wmma::row_major> "
                  f"_bf[{m['FN']}];")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm) {{")
        self.indent += 1
        self.emit(f"wmma::load_matrix_sync(_af[_fm], &{As}[(_wm * {m['FM']} + _fm) * 16 * {lda} "
                  f"+ _kk * {KF}], {lda});")
        if ft == "wmma::precision::tf32":
            self.emit("#pragma unroll")
            self.emit("for (int _e = 0; _e < _af[_fm].num_elements; ++_e) "
                      "_af[_fm].x[_e] = wmma::__float_to_tf32(_af[_fm].x[_e]);")
        self.end_loop()
        self.emit("#pragma unroll")
        self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn) {{")
        self.indent += 1
        self.emit(f"wmma::load_matrix_sync(_bf[_fn], &{Bs}[_kk * {KF} * {ldb} "
                  f"+ (_wn * {m['FN']} + _fn) * 16], {ldb});")
        if ft == "wmma::precision::tf32":
            self.emit("#pragma unroll")
            self.emit("for (int _e = 0; _e < _bf[_fn].num_elements; ++_e) "
                      "_bf[_fn].x[_e] = wmma::__float_to_tf32(_bf[_fn].x[_e]);")
        self.end_loop()
        self.emit("#pragma unroll")
        self.emit(f"for (int _fm = 0; _fm < {m['FM']}; ++_fm)")
        self.emit("#pragma unroll")
        self.emit(f"for (int _fn = 0; _fn < {m['FN']}; ++_fn)")
        self.emit(f"  wmma::mma_sync({accv}[_fm][_fn], _af[_fm], _bf[_fn], "
                  f"{accv}[_fm][_fn]);")
        self.end_loop()
        self.end_loop()

    def _stage_chunk_async(self, lazy, rows, cols, ld, buf, ck, CK, by_rows):
        """cp.async one K-chunk of a deferred load into its smem tile.

        Chunk ck covers columns [ck*CK, (ck+1)*CK) of A (by_rows=False) or
        rows of B (by_rows=True); ck=None stages the whole tile. Groups with
        contiguous, aligned, fully valid lanes go through
        __pipeline_memcpy_async; the rest copy synchronously with mask/other
        semantics.
        """
        ptr, mask, other = lazy.meta["ptr"], lazy.meta["mask"], lazy.meta["other"]
        elem = lazy.dtype
        esz = elem.itemsize
        V = self.VEC
        gb = V * esz
        align = min(gb, 16)
        numel = rows * cols
        S = self.slots(numel)

        def mask_at(idx):
            if mask is None:
                return None
            if mask.layout == CYCLIC:
                return f"{mask.var}[{idx}]"
            if mask.layout == UNIFORM:
                return mask.var
            return self.literal(mask.pyval, tp.int1)

        def other_at(idx):
            if other is None:
                return self.literal(0, elem)
            if other.layout == CONSTEXPR:
                return self.literal(other.pyval, elem)
            if other.layout == UNIFORM:
                return self.convert(other.var, other.dtype, elem)
            return self.convert(f"{other.var}[{idx}]", other.dtype, elem)

        self.emit("#pragma unroll")
        self.emit(f"for (int _g = 0; _g < {S // V}; ++_g) {{")
        self.indent += 1
        self.emit(f"int _j0 = (_g * {self.T} + _tid) * {V};")
        outer = []
        if numel < self.T * V:
            outer.append(f"_j0 < {numel}")
        self.emit(f"int _row = _j0 / {cols}, _col = _j0 % {cols};")
        sel = "_row" if by_rows else "_col"
        if ck is not None:
            outer.append(f"{sel} / {CK} == {ck}")
        self.emit(f"if ({' && '.join(outer) if outer else 'true'}) {{")
        self.indent += 1
        self.emit(f"int _i0 = _g * {V};")
        self.emit(f"int _o0 = {ptr.var}[_i0];")
        conds = [" && ".join(f"{ptr.var}[_i0 + {v}] == _o0 + {v}" for v in range(1, V))]
        conds.append(f"((((unsigned long long)(&{ptr.base}[_o0])) & {align - 1}) == 0)")
        if mask is not None:
            conds.append("(" + " && ".join(
                str(mask_at(f"_i0 + {v}")) for v in range(V)) + ")")
        self.emit(f"{elem.ctype}* _dst = &{buf}[_row * {ld} + _col];")
        self.emit(f"if ({' && '.join(conds)}) {{")
        self.indent += 1
        if gb <= 16:
            self.emit(f"__pipeline_memcpy_async(_dst, &{ptr.base}[_o0], {gb});")
        else:  # 32-byte group -> two 16-byte copies
            self.emit(f"__pipeline_memcpy_async(_dst, &{ptr.base}[_o0], 16);")
            self.emit(f"__pipeline_memcpy_async((void*)((char*)_dst + 16), "
                      f"(const void*)((const char*)(&{ptr.base}[_o0]) + 16), 16);")
        self.indent -= 1
        self.emit("} else {")
        self.indent += 1
        self.emit("#pragma unroll")
        self.emit(f"for (int _v = 0; _v < {V}; ++_v) {{")
        self.indent += 1
        self.emit("int _s = _i0 + _v; int _jv = _j0 + _v; (void)_s;")
        lane_conds = []
        if numel < self.T * V:
            lane_conds.append(f"_jv < {numel}")
        m_lane = mask_at("_s")
        src = f"{ptr.base}[{ptr.var}[_s]]"
        if m_lane is not None:
            src = f"(({m_lane}) ? {src} : {other_at('_s')})"
        line = f"{buf}[(_jv / {cols}) * {ld} + (_jv % {cols})] = {src};"
        self.emit(f"if ({' && '.join(lane_conds)}) {line}" if lane_conds else line)
        self.end_loop()
        self.end_loop()
        self.end_loop()
        self.end_loop()

    # -- assembly ----------------------------------------------------------------

    def assemble(self):
        hdr = ["// generated by newt"]
        if "fp16" in self.uses:
            hdr.append("#include <cuda_fp16.h>")
        if "bf16" in self.uses:
            hdr.append("#include <cuda_bf16.h>")
        if "mma" in self.uses:
            hdr.append("#include <mma.h>")
            hdr.append("using namespace nvcuda;")
        if "pipeline" in self.uses:
            hdr.append("#include <cuda_pipeline_primitives.h>")
        if "vec" in self.uses:
            hdr.append("template<class T, int N, int A> struct alignas(A) _nv { T d[N]; };")
        if "atomicmaxf" in self.uses:
            hdr.append(
                "__device__ inline float _nv_atomic_max_f32(float* a, float v) {\n"
                "  int old = __float_as_int(*a);\n"
                "  while (__int_as_float(old) < v) {\n"
                "    int assumed = old;\n"
                "    old = atomicCAS((int*)a, assumed, __float_as_int(v));\n"
                "    if (old == assumed) break;\n"
                "  }\n"
                "  return __int_as_float(old);\n"
                "}")
        params = []
        for name, v in self._param_order:
            if isinstance(v.dtype, tp.pointer_type):
                params.append(f"{v.dtype.element.ctype}* __restrict__ {v.var}")
                if v.dtype.element == tp.float16:
                    self.uses.add("fp16")
                if v.dtype.element == tp.bfloat16:
                    self.uses.add("bf16")
            else:
                params.append(f"{v.dtype.ctype} {v.var}")
        # re-check includes after params
        if "fp16" in self.uses and "#include <cuda_fp16.h>" not in hdr:
            hdr.insert(1, "#include <cuda_fp16.h>")
        if "bf16" in self.uses and "#include <cuda_bf16.h>" not in hdr:
            hdr.insert(1, "#include <cuda_bf16.h>")
        # pipelined-dot state lives at function scope, before any loops
        if self._prologue_decls:
            self.lines[self._prologue_idx:self._prologue_idx] = [
                "  " + d for d in self._prologue_decls]
        # smem layout: [scratch arena][ring buffers...]; rings are dedicated
        # so intervening smem ops can't clobber tiles that are still in flight
        scratch = (self.smem_bytes + 15) // 16 * 16
        base = scratch
        for i, rbytes in enumerate(self.ring_reservations):
            hdr.append(f"#define _NRB{i} {base}")
            base += rbytes
        total_smem = base
        if total_smem > MAX_SMEM:
            raise CompileError(
                f"kernel needs {total_smem} bytes of shared memory (max {MAX_SMEM}); "
                f"use smaller block sizes")
        body = []
        body.append(f'extern "C" __global__ void __launch_bounds__({self.T}) '
                    f"{self.name}({', '.join(params)}) {{")
        if total_smem > 0:
            body.append("  extern __shared__ __align__(16) char _smem[];")
        body += self.lines
        body.append("}")
        return "\n".join(hdr + [""] + body) + "\n", total_smem


def compile_fn(fndef, fn_globals, param_values, constexprs, num_warps, kernel_name):
    """param_values: list of (python-name, Value with C var already set)."""
    cg = Codegen(fndef, fn_globals, param_values, constexprs, num_warps, kernel_name)
    cg._param_order = param_values
    src, smem = cg.compile()
    return src, smem
