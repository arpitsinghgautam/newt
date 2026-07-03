"""deuteron compiler: tile-DSL AST -> newt kernel source.

Like Helion emits Triton code, deuteron emits newt code. The generated source
is a plain @newt.jit kernel with every tile size a constexpr parameter, so
one source string serves every autotuner config.
"""

import ast

from . import language as dl


class TraceError(Exception):
    pass


class TileVar:
    def __init__(self, name, size_sym, kind, block):
        self.name = name          # python-level tile variable name
        self.size_sym = size_sym  # runtime size symbol, e.g. "x_d1"
        self.kind = kind          # 'grid' | 'loop' | 'full'
        self.block = block        # constexpr name, e.g. "BLOCK_tm"

    @property
    def offs(self):
        return f"offs_{self.name}"

    @property
    def mask(self):
        return f"mask_{self.name}"


class Expr:
    def __init__(self, code, mask=None, ndim=0):
        self.code = code
        self.mask = mask  # mask expression covering padded lanes, or None
        self.ndim = ndim


class KernelSource:
    def __init__(self, name, source, params, tile_vars, grid_tiles, tunable, store_params):
        self.name = name
        self.source = source          # generated newt module source
        self.params = params          # launch-arg spec: list of ('tensor', pyname) / ('scalar', pyname)
        self.tile_vars = tile_vars
        self.grid_tiles = grid_tiles  # TileVars forming the launch grid
        self.tunable = tunable        # constexpr names the autotuner may vary
        self.store_params = store_params  # tensor param names written by the kernel


class SourceGen(ast.NodeVisitor):
    def __init__(self, fn, fndef, tensor_params, scalar_params, dtypes):
        self.fn = fn
        self.fndef = fndef
        self.tensor_params = tensor_params  # name -> ndim
        self.scalar_params = scalar_params  # [name]
        self.dtypes = dtypes                # name -> torch dtype
        self.lines = []
        self.indent = 1
        self.tiles = {}       # tile var name -> TileVar
        self.full_tiles = {}  # size_sym -> TileVar
        self.grid_tiles = []
        self.locals = {}      # local name -> Expr
        self.has_dot = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult)
                           for n in ast.walk(fndef))
        self.saw_grid = False
        self.counter = 0

    def err(self, node, msg):
        raise TraceError(f"{self.fndef.name}:{getattr(node, 'lineno', '?')}: {msg}")

    def emit(self, s):
        self.lines.append("    " * self.indent + s)

    # -- tile loops ------------------------------------------------------------

    def _tile_sizes(self, node):
        """Resolve dt.tile(...) argument into a list of size symbols."""
        if not (isinstance(node, ast.Call) and self._is_dt(node.func, "tile")):
            self.err(node, "expected dt.tile(...)")
        if len(node.args) != 1:
            self.err(node, "dt.tile takes one argument")
        arg = node.args[0]
        elts = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
        return [self._size_symbol(e) for e in elts]

    def _size_symbol(self, node):
        """A dimension size expression -> runtime symbol string."""
        # x.shape[i]
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr in ("shape",)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self.tensor_params):
            idx = node.slice
            if not isinstance(idx, ast.Constant):
                self.err(node, "shape index must be a literal")
            return f"{node.value.value.id}_d{idx.value}"
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return str(node.value)
        if isinstance(node, ast.Name) and node.id in self.scalar_params:
            return node.id
        self.err(node, "tile sizes must be x.shape[i], an int literal, or a scalar param")

    def visit_For(self, node):
        if node.orelse:
            self.err(node, "for/else not supported")
        sizes = self._tile_sizes(node.iter)
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        if len(targets) != len(sizes):
            self.err(node, f"{len(targets)} tile vars but {len(sizes)} sizes")
        names = []
        for t in targets:
            if not isinstance(t, ast.Name):
                self.err(node, "tile targets must be plain names")
            names.append(t.id)
        if not self.saw_grid:
            # outermost tile loop -> launch grid
            self.saw_grid = True
            if len(names) > 3:
                self.err(node, "grid tile loops support at most 3 dimensions")
            for i, (nm, sym) in enumerate(zip(names, sizes)):
                tv = TileVar(nm, sym, "grid", f"BLOCK_{nm.upper()}")
                self.tiles[nm] = tv
                self.grid_tiles.append(tv)
                self.emit(f"pid_{i} = nl.program_id({i})")
                self.emit(f"{tv.offs} = pid_{i} * {tv.block} + nl.arange(0, {tv.block})")
                self.emit(f"{tv.mask} = {tv.offs} < {sym}")
            for stmt in node.body:
                self.visit_stmt(stmt)
        else:
            if len(names) != 1:
                self.err(node, "inner tile loops iterate one dimension at a time")
            nm, sym = names[0], sizes[0]
            tv = TileVar(nm, sym, "loop", f"BLOCK_{nm.upper()}")
            self.tiles[nm] = tv
            self.emit(f"for _i_{nm} in range(0, nl.cdiv({sym}, {tv.block})):")
            self.indent += 1
            self.emit(f"{tv.offs} = _i_{nm} * {tv.block} + nl.arange(0, {tv.block})")
            self.emit(f"{tv.mask} = {tv.offs} < {sym}")
            for stmt in node.body:
                self.visit_stmt(stmt)
            self.indent -= 1

    # -- statements --------------------------------------------------------------

    def visit_stmt(self, stmt):
        if isinstance(stmt, ast.For):
            return self.visit_For(stmt)
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                self.err(stmt, "single assignment targets only")
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Subscript):
                return self._store(stmt, tgt, self.expr(stmt.value))
            if isinstance(tgt, ast.Name):
                e = self.expr(stmt.value)
                self.locals[tgt.id] = Expr(tgt.id, e.mask, e.ndim)
                self.emit(f"{tgt.id} = {e.code}")
                return
            self.err(stmt, "unsupported assignment target")
        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                self.err(stmt, "augmented assignment needs a plain name target")
            name = stmt.target.id
            if name not in self.locals:
                self.err(stmt, f"'{name}' not defined before augmented assignment")
            cur = self.locals[name]
            # acc += a @ b  ->  fused into the dot accumulator
            if isinstance(stmt.op, ast.Add) and isinstance(stmt.value, ast.BinOp) \
                    and isinstance(stmt.value.op, ast.MatMult):
                a = self.expr(stmt.value.left)
                b = self.expr(stmt.value.right)
                self.emit(f"{name} = nl.dot({a.code}, {b.code}, {name})")
                self.locals[name] = Expr(name, None, 2)
                return
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            sym = ops.get(type(stmt.op))
            if sym is None:
                self.err(stmt, "unsupported augmented op")
            e = self.expr(stmt.value)
            self.emit(f"{name} = {name} {sym} ({e.code})")
            self.locals[name] = Expr(name, cur.mask, max(cur.ndim, e.ndim))
            return
        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant):
                return  # stray docstring
            self.err(stmt, "expression statements have no effect in deuteron kernels")
        self.err(stmt, f"unsupported statement: {type(stmt).__name__}")

    # -- tensor access ---------------------------------------------------------

    def _index_tiles(self, node, sub):
        """Resolve subscript indices on a tensor param into TileVars."""
        pname = sub.value.id
        ndim = self.tensor_params[pname]
        idx = sub.slice
        elts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
        if len(elts) != ndim:
            self.err(node, f"'{pname}' has {ndim} dims but {len(elts)} indices given")
        tiles = []
        for d, e in enumerate(elts):
            if isinstance(e, ast.Name) and e.id in self.tiles:
                tiles.append(self.tiles[e.id])
            elif isinstance(e, ast.Slice) and e.lower is None and e.upper is None:
                sym = f"{pname}_d{d}"
                tv = self.full_tiles.get(sym)
                if tv is None:
                    fid = len(self.full_tiles)
                    tv = TileVar(f"full{fid}", sym, "full", f"BLOCK_FULL{fid}")
                    self.full_tiles[sym] = tv
                tiles.append(tv)
            else:
                self.err(node, "tensor indices must be tile variables or ':'")
        return pname, tiles

    def _ptr_and_mask(self, node, sub):
        pname, tiles = self._index_tiles(node, sub)
        if len(tiles) == 1:
            t = tiles[0]
            return pname, f"{pname}_ptr + {t.offs} * {pname}_s0", t.mask, 1
        if len(tiles) == 2:
            a, b = tiles
            ptr = (f"{pname}_ptr + {a.offs}[:, None] * {pname}_s0 "
                   f"+ {b.offs}[None, :] * {pname}_s1")
            mask = f"{a.mask}[:, None] & {b.mask}[None, :]"
            return pname, ptr, mask, 2
        self.err(node, "only 1D/2D tensor access is supported")

    def _store(self, node, tgt, value):
        if not (isinstance(tgt.value, ast.Name) and tgt.value.id in self.tensor_params):
            self.err(node, "stores must target a tensor parameter")
        pname, ptr, mask, _ = self._ptr_and_mask(node, tgt)
        self._stores.add(pname)
        self.emit(f"nl.store({ptr}, {value.code}, mask={mask})")

    # -- expressions --------------------------------------------------------------

    _BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
    _CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">="}

    def expr(self, node):
        if isinstance(node, ast.Constant):
            return Expr(repr(node.value))
        if isinstance(node, ast.Name):
            if node.id in self.locals:
                return self.locals[node.id]
            if node.id in self.scalar_params:
                return Expr(node.id)
            if node.id in self.tiles:
                self.err(node, f"tile '{node.id}' can only be used as an index or size")
            self.err(node, f"unknown name '{node.id}'")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            e = self.expr(node.operand)
            return Expr(f"(-{e.code})", e.mask, e.ndim)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.MatMult):
                a, b = self.expr(node.left), self.expr(node.right)
                return Expr(f"nl.dot({a.code}, {b.code})", None, 2)
            sym = self._BIN.get(type(node.op))
            if sym is None:
                self.err(node, f"unsupported operator {type(node.op).__name__}")
            a, b = self.expr(node.left), self.expr(node.right)
            return Expr(f"({a.code} {sym} {b.code})", a.mask or b.mask,
                        max(a.ndim, b.ndim))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.err(node, "chained comparisons not supported")
            sym = self._CMP.get(type(node.ops[0]))
            a = self.expr(node.left)
            b = self.expr(node.comparators[0])
            return Expr(f"({a.code} {sym} {b.code})", a.mask or b.mask,
                        max(a.ndim, b.ndim))
        if isinstance(node, ast.Subscript):
            return self._subscript_expr(node)
        if isinstance(node, ast.Call):
            return self._call_expr(node)
        self.err(node, f"unsupported expression: {type(node).__name__}")

    def _subscript_expr(self, node):
        # tensor load
        if isinstance(node.value, ast.Name) and node.value.id in self.tensor_params:
            pname, ptr, mask, ndim = self._ptr_and_mask(node, node)
            import torch

            other = "0" if self.dtypes[pname] in (torch.int32, torch.int64) else "0.0"
            return Expr(f"nl.load({ptr}, mask={mask}, other={other})", mask, ndim)
        # x.shape[i] as a runtime scalar
        if (isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in self.tensor_params):
            return Expr(self._size_symbol(node))
        # m[:, None] style expansion on locals / computed values
        base = self.expr(node.value)
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        parts = []
        added = 0
        for e in elts:
            if isinstance(e, ast.Constant) and e.value is None:
                parts.append("None")
                added += 1
            elif isinstance(e, ast.Slice) and e.lower is None and e.upper is None:
                parts.append(":")
            else:
                self.err(node, "only ':' and None subscripts on computed values")
        mask = base.mask
        if mask and added:
            # align the mask with the value's new dims
            mask = f"({mask})[{', '.join(parts)}]"
        return Expr(f"({base.code})[{', '.join(parts)}]", mask, base.ndim + added)

    def _is_dt(self, func, name=None):
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            mod = self.fn.__globals__.get(func.value.id)
            if mod is not None and getattr(mod, "__name__", "").endswith("deuteron.language") or \
               getattr(mod, "__name__", "") == "deuteron":
                return name is None or func.attr == name
        return False

    def _call_expr(self, node):
        # dt.* functions
        if isinstance(node.func, ast.Attribute) and self._is_dt(node.func):
            fname = node.func.attr
            if fname == "zeros" or fname == "full":
                return self._zeros_expr(node, fname)
            tmpl = dl.NEWT_FNS.get(fname)
            if tmpl is None:
                self.err(node, f"dt.{fname} is not supported in compiled kernels")
            args = [self.expr(a) for a in node.args]
            code = tmpl.format(*[a.code for a in args])
            mask = next((a.mask for a in args if a.mask), None)
            return Expr(code, mask, max((a.ndim for a in args), default=0))
        # float('-inf') and friends
        if isinstance(node.func, ast.Name) and node.func.id in ("float", "int"):
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                return Expr(f"{node.func.id}({arg.value!r})")
        # reduction / cast methods on values
        if isinstance(node.func, ast.Attribute):
            mname = node.func.attr
            base = self.expr(node.func.value)
            if mname in ("sum", "max", "min", "amax", "amin", "mean"):
                return self._reduce_expr(node, base, mname)
            if mname == "to":
                self.err(node, "use dt.zeros(..., dtype=...) / implicit store casts instead of .to()")
        self.err(node, f"unsupported call {ast.dump(node.func)[:60]}")

    def _zeros_expr(self, node, fname):

        shape_arg = node.args[0]
        elts = shape_arg.elts if isinstance(shape_arg, (ast.List, ast.Tuple)) else [shape_arg]
        blocks = []
        for e in elts:
            if isinstance(e, ast.Name) and e.id in self.tiles:
                blocks.append(self.tiles[e.id].block)
            elif isinstance(e, ast.Constant):
                blocks.append(str(e.value))
            else:
                self.err(node, "dt.zeros shape entries must be tile vars or int literals")
        dtype = "nl.float32"
        for kw in node.keywords:
            if kw.arg == "dtype":
                val = self._const_attr(kw.value)
                dtype = dl.DTYPE_NAMES.get(val)
                if dtype is None:
                    self.err(node, "unsupported dtype")
        if fname == "zeros":
            return Expr(f"nl.zeros(({', '.join(blocks)},), dtype={dtype})", None, len(blocks))
        fill = self.expr(node.args[1])
        return Expr(f"nl.full(({', '.join(blocks)},), {fill.code}, dtype={dtype})",
                    None, len(blocks))

    def _const_attr(self, node):
        """Evaluate dt.float32 / torch.float32-style dtype references."""
        try:
            code = compile(ast.Expression(node), "<dtype>", "eval")
            return eval(code, self.fn.__globals__)
        except Exception:
            return None

    def _reduce_expr(self, node, base, mname):
        axis = None
        if node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant):
                axis = a.value
        for kw in node.keywords:
            if kw.arg in ("axis", "dim") and isinstance(kw.value, ast.Constant):
                axis = kw.value.value
        if axis is not None and axis < 0:
            if base.ndim <= 0:
                self.err(node, "cannot resolve negative axis; use a non-negative one")
            axis = base.ndim + axis
        if mname == "mean":
            inner = base.code
            if base.mask:
                inner = f"nl.where({base.mask}, {base.code}, 0.0)"
            ax = "" if axis is None else f", axis={axis}"
            # divide by the true (unpadded) reduced size
            size = self._reduced_size(node, base, axis)
            return Expr(f"(nl.sum({inner}{ax}) / {size})", None,
                        max(0, base.ndim - 1) if axis is not None else 0)
        fn, fill = dl.REDUCTIONS[mname]
        inner = base.code
        if base.mask:
            inner = f"nl.where({base.mask}, {base.code}, {fill})"
        ax = "" if axis is None else f", axis={axis}"
        ndim = 0 if axis is None else max(0, base.ndim - 1)
        return Expr(f"{fn}({inner}{ax})", None, ndim)

    def _reduced_size(self, node, base, axis):
        self.err(node, "mean() is not supported yet; use .sum(axis) / explicit size")


def generate(fn, fndef, args):
    """Trace fn(args) -> KernelSource."""
    import torch

    params = [a.arg for a in fndef.args.args]
    if len(params) != len(args):
        raise TraceError(f"{fndef.name}: expected {len(params)} args, got {len(args)}")
    tensor_params = {}
    scalar_params = []
    dtypes = {}
    for name, val in zip(params, args):
        if torch.is_tensor(val):
            tensor_params[name] = val.dim()
            dtypes[name] = val.dtype
        elif isinstance(val, (int, float)):
            scalar_params.append(name)
        else:
            raise TraceError(f"unsupported argument type for '{name}': {type(val)}")

    g = SourceGen(fn, fndef, tensor_params, scalar_params, dtypes)
    g._stores = set()
    body = fndef.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    for stmt in body:
        if isinstance(stmt, ast.For):
            g.visit_For(stmt)
        else:
            g.err(stmt, "top level of a deuteron kernel must be `for ... in dt.tile(...)`")

    # prologue for full-dimension tiles (must precede the body that uses them,
    # but they're discovered during body generation -> prepend)
    full_lines = []
    for tv in g.full_tiles.values():
        full_lines.append("    " + f"{tv.offs} = nl.arange(0, {tv.block})")
        full_lines.append("    " + f"{tv.mask} = {tv.offs} < {tv.size_sym}")

    # signature: tensors (ptr + dims + strides), scalars, constexpr blocks
    sig = []
    launch_params = []
    for name, ndim in tensor_params.items():
        sig.append(f"{name}_ptr")
        launch_params.append(("tensor", name))
        for d in range(ndim):
            sig.append(f"{name}_d{d}")
        for d in range(ndim):
            sig.append(f"{name}_s{d}")
    for name in scalar_params:
        sig.append(name)
        launch_params.append(("scalar", name))

    tile_vars = list(g.tiles.values()) + list(g.full_tiles.values())
    tunable = [tv.block for tv in g.tiles.values()]
    consts = [tv.block for tv in tile_vars]
    sig += [f"{c}: nl.constexpr" for c in consts]

    kname = f"{fndef.name}_newt"
    src = [
        "import newt",
        "import newt.language as nl",
        "",
        "@newt.jit",
        f"def {kname}({', '.join(sig)}):",
    ]
    src += full_lines
    src += g.lines
    source = "\n".join(src) + "\n"
    return KernelSource(kname, source, launch_params, tile_vars,
                        g.grid_tiles, tunable, g._stores)
