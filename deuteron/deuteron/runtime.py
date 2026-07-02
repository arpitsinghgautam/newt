"""deuteron runtime: trace -> generate newt source -> autotune -> launch.

Autotuning protocol (mini-Helion):
  1. run the kernel function *eagerly* (full-size tiles = plain PyTorch) on
     cloned inputs -> the correctness oracle
  2. generate newt source once; tile sizes are constexprs, so every config
     reuses the same source
  3. evaluate candidate configs: reject wrong-output ones, time the rest
     (random sample + local pattern search around the best)
  4. persist the winner keyed by (source, shape bucket, dtypes)
"""

import ast
import hashlib
import inspect
import itertools
import json
import linecache
import os
import random
import textwrap

import torch

from . import codegen
from .language import EagerTensor


def _cache_path():
    d = os.environ.get("DEUTERON_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".deuteron")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "configs.json")


def _next_pow2(n):
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class Config(dict):
    @property
    def num_warps(self):
        return self.get("num_warps", 4)


class Kernel:
    def __init__(self, fn, config=None, autotune=True, samples=12, verbose=False):
        self.fn = fn
        self.__name__ = fn.__name__
        src = textwrap.dedent(inspect.getsource(fn))
        self.fndef = ast.parse(src).body[0]
        self.fixed_config = Config(config) if config else None
        self.autotune_enabled = autotune and config is None
        self.samples = samples
        self.verbose = verbose or bool(os.environ.get("DEUTERON_VERBOSE"))
        self._ks = None          # KernelSource (depends on arg ndims/dtypes)
        self._ks_sig = None
        self._jit = None
        self._config_cache = {}  # key -> Config

    # -- public API ------------------------------------------------------------

    def __call__(self, *args):
        if os.environ.get("DEUTERON_INTERPRET"):
            return self._eager(args)
        ks = self._source_for(args)
        cfg = self.fixed_config or self._get_config(ks, args)
        self._launch(ks, args, cfg)

    def ref(self, *args):
        """Run the eager (pure PyTorch) reference implementation."""
        return self._eager(args)

    def to_newt_source(self, *args):
        """The generated newt kernel source for these example arguments."""
        return self._source_for(args).source

    @property
    def best_config(self):
        vals = list(self._config_cache.values())
        return vals[-1] if vals else self.fixed_config

    # -- internals ----------------------------------------------------------------

    def _eager(self, args):
        wrapped = [EagerTensor(a) if torch.is_tensor(a) else a for a in args]
        return self.fn(*wrapped)

    def _arg_sig(self, args):
        return tuple(
            (a.dim(), str(a.dtype)) if torch.is_tensor(a) else type(a).__name__
            for a in args
        )

    def _source_for(self, args):
        sig = self._arg_sig(args)
        if self._ks is None or self._ks_sig != sig:
            self._ks = codegen.generate(self.fn, self.fndef, args)
            self._ks_sig = sig
            ns = {}
            fname = f"<deuteron:{self.__name__}>"
            # register with linecache so newt's inspect.getsource() works
            linecache.cache[fname] = (
                len(self._ks.source), None, self._ks.source.splitlines(True), fname)
            exec(compile(self._ks.source, fname, "exec"), ns)
            self._jit = ns[self._ks.name]
            if self.verbose:
                print(f"=== deuteron: generated newt source for {self.__name__} ===")
                print(self._ks.source)
        return self._ks

    def _bind_sizes(self, ks, args):
        """Map size symbols (x_d0, ...) to concrete values for these args."""
        params = [a.arg for a in self.fndef.args.args]
        by_name = dict(zip(params, args))
        sizes = {}
        for kind, name in ks.params:
            if kind == "tensor":
                t = by_name[name]
                for d in range(t.dim()):
                    sizes[f"{name}_d{d}"] = t.shape[d]
            else:
                sizes[name] = by_name[name]
        return sizes

    def _launch_args(self, ks, args):
        params = [a.arg for a in self.fndef.args.args]
        by_name = dict(zip(params, args))
        out = []
        for kind, name in ks.params:
            v = by_name[name]
            if kind == "tensor":
                out.append(v)
                out.extend(int(s) for s in v.shape)
                out.extend(int(s) for s in v.stride())
            else:
                out.append(v)
        return out

    def _resolve_size(self, sizes, sym):
        return int(sizes[sym]) if sym in sizes else int(sym)

    def _constexprs(self, ks, sizes, cfg):
        kw = {}
        for tv in ks.tile_vars:
            if tv.kind == "full":
                kw[tv.block] = _next_pow2(self._resolve_size(sizes, tv.size_sym))
            else:
                kw[tv.block] = cfg[tv.block]
        return kw

    def _launch(self, ks, args, cfg):
        sizes = self._bind_sizes(ks, args)
        kw = self._constexprs(ks, sizes, cfg)
        grid = tuple(
            -(-self._resolve_size(sizes, tv.size_sym) // kw[tv.block])
            for tv in ks.grid_tiles
        )
        self._jit[grid](*self._launch_args(ks, args), **kw, num_warps=cfg.num_warps)

    # -- autotuning ---------------------------------------------------------------

    def _key(self, ks, args):
        h = hashlib.sha256(ks.source.encode()).hexdigest()[:12]
        shapes = tuple(
            tuple(_next_pow2(s) for s in a.shape) if torch.is_tensor(a) else None
            for a in args
        )
        dts = tuple(str(a.dtype) if torch.is_tensor(a) else "" for a in args)
        return f"{self.__name__}:{h}:{shapes}:{dts}"

    def _get_config(self, ks, args):
        key = self._key(ks, args)
        cfg = self._config_cache.get(key)
        if cfg is not None:
            return cfg
        disk = {}
        try:
            with open(_cache_path()) as f:
                disk = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        if key in disk:
            cfg = Config(disk[key])
            self._config_cache[key] = cfg
            return cfg
        cfg = self._autotune(ks, args)
        self._config_cache[key] = cfg
        disk[key] = dict(cfg)
        try:
            with open(_cache_path(), "w") as f:
                json.dump(disk, f, indent=1)
        except OSError:
            pass
        return cfg

    def _space(self, ks):
        blocks = [16, 32, 64, 128] if any(
            "nl.dot(" in ln for ln in ks.source.splitlines()
        ) else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        space = {name: blocks for name in ks.tunable}
        space["num_warps"] = [2, 4, 8]
        return space

    def _neighbors(self, cfg, space):
        out = []
        for k, options in space.items():
            i = options.index(cfg[k]) if cfg[k] in options else 0
            for j in (i - 1, i + 1):
                if 0 <= j < len(options):
                    n = Config(cfg)
                    n[k] = options[j]
                    out.append(n)
        return out

    def _autotune(self, ks, args):
        from newt.testing import do_bench

        space = self._space(ks)
        keys = list(space)

        # oracle: eager run on clones
        clones = [a.clone() if torch.is_tensor(a) else a for a in args]
        self._eager(clones)
        params = [a.arg for a in self.fndef.args.args]
        by_name = dict(zip(params, clones))
        expected = {p: by_name[p].clone() for p in ks.store_params}

        work = [a.clone() if torch.is_tensor(a) else a for a in args]
        by_name_w = dict(zip(params, work))
        outs = {p: by_name_w[p] for p in ks.store_params}
        originals = {p: dict(zip(params, args))[p] for p in ks.store_params}

        def tol(t):
            if t.dtype in (torch.float16, torch.bfloat16):
                return 2e-2
            return 2e-2 if "nl.dot(" in ks.source else 1e-3

        def try_cfg(cfg):
            for p, o in outs.items():
                o.copy_(originals[p])  # reset outputs between candidates
            try:
                self._launch(ks, work, cfg)
                torch.cuda.synchronize()
            except Exception as e:
                if self.verbose:
                    print(f"  {dict(cfg)}: invalid ({str(e)[:80]})")
                return None
            for p, exp in expected.items():
                t = tol(exp)
                if not torch.allclose(outs[p].float(), exp.float(), rtol=t, atol=t):
                    if self.verbose:
                        print(f"  {dict(cfg)}: WRONG RESULT (skipped)")
                    return None
            ms = do_bench(lambda: self._launch(ks, work, cfg), warmup=3, rep=10)
            if self.verbose:
                print(f"  {dict(cfg)}: {ms:.4f} ms")
            return ms

        # candidates: full grid if small, else seeded random sample
        all_cfgs = [Config(zip(keys, vals)) for vals in itertools.product(
            *[space[k] for k in keys])]
        rng = random.Random(0)
        cands = all_cfgs if len(all_cfgs) <= self.samples else rng.sample(
            all_cfgs, self.samples)

        seen, results = set(), []
        for cfg in cands:
            t = frozenset(cfg.items())
            if t in seen:
                continue
            seen.add(t)
            ms = try_cfg(cfg)
            if ms is not None:
                results.append((ms, cfg))
        if not results:
            raise RuntimeError(
                f"deuteron: no valid config found for {self.__name__} "
                f"(try DEUTERON_VERBOSE=1)")
        results.sort(key=lambda r: r[0])
        best_ms, best = results[0]

        # local pattern search
        for _ in range(3):
            improved = False
            for n in self._neighbors(best, space):
                t = frozenset(n.items())
                if t in seen:
                    continue
                seen.add(t)
                ms = try_cfg(n)
                if ms is not None and ms < best_ms:
                    best_ms, best = ms, n
                    improved = True
            if not improved:
                break
        if self.verbose:
            print(f"deuteron: {self.__name__} best {dict(best)} = {best_ms:.4f} ms")
        return best


def kernel(fn=None, *, config=None, autotune=True, samples=12, verbose=False):
    """@deuteron.kernel - compile a tile-DSL function to autotuned newt kernels."""
    if fn is None:
        return lambda f: Kernel(f, config=config, autotune=autotune,
                                samples=samples, verbose=verbose)
    return Kernel(fn)
