"""@newt.jit: compile-on-first-call kernel wrapper.

Specialization: a kernel is recompiled per unique combination of
(constexpr values, argument type signature, num_warps) - same policy as
Triton minus its alignment specialization.
"""

import ast
import ctypes
import functools
import hashlib
import inspect
import os
import textwrap

from ..compiler import codegen
from ..compiler import types as tp
from . import cuda as _cuda


def _cache_dir():
    d = os.environ.get("NEWT_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".newt", "cache")
    os.makedirs(d, exist_ok=True)
    return d


@functools.cache
def _arch():
    import torch

    cap = torch.cuda.get_device_capability()
    return f"sm_{cap[0]}{cap[1]}"


def _is_constexpr_ann(ann):
    from .. import language as nl

    if ann is nl.constexpr or isinstance(ann, nl.constexpr):
        return True
    return isinstance(ann, str) and "constexpr" in ann


class CompiledKernel:
    def __init__(self, name, source, smem_bytes, num_warps, param_names):
        self.name = name
        self.source = source
        self.smem_bytes = smem_bytes
        self.num_warps = num_warps
        self.param_names = param_names
        key = hashlib.sha256((source + _arch() + "v1").encode()).hexdigest()[:32]
        path = os.path.join(_cache_dir(), f"{key}.cubin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                cubin = f.read()
        else:
            try:
                cubin = _cuda.compile_cuda(source, name, _arch())
            except _cuda.NVRTCError as e:
                lined = "\n".join(
                    f"{i+1:4d} | {ln}" for i, ln in enumerate(source.splitlines()))
                raise _cuda.NVRTCError(f"{e}\n--- generated source ---\n{lined}") from None
            with open(path, "wb") as f:
                f.write(cubin)
            with open(os.path.join(_cache_dir(), f"{key}.cu"), "w") as f:
                f.write(source)
        self.kernel = _cuda.Kernel(cubin, name, smem_bytes)

    def launch(self, grid, cargs, stream):
        self.kernel.launch(grid, (self.num_warps * 32,), cargs, stream)


class JITFunction:
    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        self.fndef = tree.body[0]
        assert isinstance(self.fndef, ast.FunctionDef)
        sig = inspect.signature(fn)
        self.param_names = list(sig.parameters)
        self.constexpr_names = [
            n for n, p in sig.parameters.items() if _is_constexpr_ann(p.annotation)
        ]
        self.defaults = {
            n: p.default for n, p in sig.parameters.items() if p.default is not inspect.Parameter.empty
        }
        self.cache = {}

    def __getitem__(self, grid):
        return functools.partial(self._run, grid)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"@newt.jit kernel '{self.fn.__name__}' must be launched with a grid: "
            f"kernel[grid](...)"
        )

    def _classify(self, args, kwargs):
        """Returns (constexprs dict, runtime {name: value})."""
        bound = {}
        for name, val in zip(self.param_names, args):
            bound[name] = val
        for name, val in kwargs.items():
            if name in self.param_names:
                if name in bound:
                    raise TypeError(f"duplicate argument {name}")
                bound[name] = val
        if len(args) > len(self.param_names):
            raise TypeError(
                f"{self.fn.__name__}() takes {len(self.param_names)} arguments "
                f"but {len(args)} were given")
        unknown = [k for k in kwargs if k not in self.param_names]
        if unknown:
            raise TypeError(f"unknown kernel arguments: {unknown}")
        for name, dflt in self.defaults.items():
            bound.setdefault(name, dflt)
        missing = [n for n in self.param_names if n not in bound]
        if missing:
            raise TypeError(f"missing kernel arguments: {missing}")
        constexprs = {n: bound[n] for n in self.constexpr_names}
        runtime = {n: bound[n] for n in self.param_names if n not in self.constexpr_names}
        return constexprs, runtime

    @staticmethod
    def _type_of(name, val):
        import torch

        if isinstance(val, torch.Tensor):
            if not val.is_cuda:
                raise TypeError(f"argument '{name}' must be a CUDA tensor")
            elem = tp.TORCH_TO_NEWT.get(str(val.dtype))
            if elem is None:
                raise TypeError(f"unsupported tensor dtype {val.dtype} for '{name}'")
            return tp.pointer_type(elem)
        if isinstance(val, bool):
            return tp.int1
        if isinstance(val, int):
            if not -(2**63) <= val < 2**63:
                raise TypeError(f"integer argument '{name}'={val} does not fit int64")
            return tp.int32 if -(2**31) <= val < 2**31 else tp.int64
        if isinstance(val, float):
            return tp.float32
        raise TypeError(f"unsupported kernel argument type {type(val)} for '{name}'")

    def _compile(self, key, constexprs, runtime_types, num_warps, num_stages=2):
        params = []
        for name in self.param_names:
            if name in self.constexpr_names:
                continue
            t = runtime_types[name]
            cname = f"g_{name}" if name in codegen._CKEYWORDS else name
            params.append((name, codegen.Value(codegen.UNIFORM, t, (), cname)))
        khash = hashlib.sha256(repr(key).encode()).hexdigest()[:8]
        kname = f"{self.fn.__name__}_{khash}"
        source, smem = codegen.compile_fn(
            self.fndef, self.fn.__globals__, params, constexprs, num_warps, kname,
            num_stages=num_stages,
        )
        if os.environ.get("NEWT_DEBUG"):
            print(f"=== newt: {kname} (smem={smem}b, warps={num_warps}) ===")
            print(source)
        return CompiledKernel(kname, source, smem, num_warps,
                              [n for n, _ in params])

    def _run(self, grid, *args, **kwargs):
        import torch

        num_warps = kwargs.pop("num_warps", 4)
        num_stages = kwargs.pop("num_stages", 3)  # dot pipeline depth (ring slots)
        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"num_warps must be a power of two <= 32, got {num_warps}")
        if not 1 <= num_stages <= 8:
            raise ValueError(f"num_stages must be in 1..8, got {num_stages}")
        constexprs, runtime = self._classify(args, kwargs)
        runtime_types = {n: self._type_of(n, v) for n, v in runtime.items()}
        env_knobs = tuple(os.environ.get(k, "") for k in (
            "NEWT_MMA", "NEWT_VEC", "NEWT_ASYNC_DOT", "NEWT_PIPELINE_DOT"))
        key = (
            tuple(sorted((k, repr(v)) for k, v in constexprs.items())),
            tuple((n, t.name) for n, t in runtime_types.items()),
            num_warps, num_stages, env_knobs,
        )
        compiled = self.cache.get(key)
        if compiled is None:
            compiled = self._compile(key, constexprs, runtime_types, num_warps,
                                     num_stages)
            self.cache[key] = compiled

        cargs = []
        for name in compiled.param_names:
            v = runtime[name]
            t = runtime_types[name]
            if isinstance(t, tp.pointer_type):
                if v.device.index not in (None, torch.cuda.current_device()):
                    raise ValueError(
                        f"tensor '{name}' is on {v.device} but the current device is "
                        f"cuda:{torch.cuda.current_device()}; newt launches on the "
                        f"current device")
                cargs.append(ctypes.c_void_p(v.data_ptr()))
            elif t == tp.int1:
                cargs.append(ctypes.c_bool(v))
            elif t == tp.int64:
                cargs.append(ctypes.c_int64(v))
            elif t == tp.int32:
                cargs.append(ctypes.c_int32(v))
            else:
                cargs.append(ctypes.c_float(v))

        if callable(grid):
            meta = dict(runtime)      # all bound args, like Triton's META
            meta.update(constexprs)
            meta["num_warps"] = num_warps
            g = grid(meta)
        else:
            g = grid
        if isinstance(g, int):
            g = (g,)
        g = tuple(int(x) for x in g)
        if any(x <= 0 for x in g):
            return  # empty grid: no-op, like Triton
        stream = torch.cuda.current_stream().cuda_stream
        compiled.launch(g, cargs, stream)


def jit(fn=None):
    if fn is None:
        return jit
    return JITFunction(fn)
