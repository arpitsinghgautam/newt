"""@newt.autotune / @newt.heuristics - same decorators as Triton."""

import builtins
import inspect


class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=1):
        self.kwargs = dict(kwargs)
        self.num_warps = num_warps
        self.num_stages = num_stages

    def all_kwargs(self):
        return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}

    def __repr__(self):
        kw = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"Config({kw}, num_warps={self.num_warps})"


class Autotuner:
    def __init__(self, jitfn, configs, key, reset_to_zero=None, warmup=5, rep=20):
        self.jitfn = jitfn
        self.configs = configs
        self.key = key
        self.reset_to_zero = reset_to_zero or []
        self.warmup = warmup
        self.rep = rep
        self.cache = {}
        self.__name__ = getattr(jitfn, "__name__", "kernel")

    def __getitem__(self, grid):
        def runner(*args, **kwargs):
            return self._run(grid, args, kwargs)

        return runner

    def _key_values(self, args, kwargs):
        names = self.jitfn.param_names
        bound = dict(zip(names, args))
        bound.update(kwargs)
        for n, d in getattr(self.jitfn, "defaults", {}).items():
            bound.setdefault(n, d)
        vals = []
        for k in self.key:
            v = bound.get(k)
            vals.append(v if not hasattr(v, "shape") else tuple(v.shape))
        # dtypes of tensor args are part of the key implicitly
        import torch

        for n, v in bound.items():
            if isinstance(v, torch.Tensor):
                vals.append(str(v.dtype))
        return tuple(vals)

    def _run(self, grid, args, kwargs):
        from ..testing import do_bench

        keyv = self._key_values(args, kwargs)
        names = self.jitfn.param_names
        bound = dict(zip(names, args))
        bound.update(kwargs)
        best = self.cache.get(keyv)
        tuned_now = False
        if best is None:
            tuned_now = True
            timings = []
            last_exc = None
            for cfg in self.configs:
                call_kwargs = {**kwargs, **cfg.kwargs,
                               "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}

                def run_cfg():
                    for name in self.reset_to_zero:
                        bound[name].zero_()
                    self.jitfn[grid](*args, **call_kwargs)

                try:
                    ms = do_bench(run_cfg, warmup=self.warmup, rep=self.rep)
                    timings.append((ms, cfg))
                except Exception as e:  # config invalid (smem/tiling/...)
                    last_exc = e
                    continue
            if not timings:
                raise RuntimeError(
                    f"autotune: no valid config for {self.__name__}") from last_exc
            timings.sort(key=lambda t: t[0])
            best = timings[0][1]
            self.cache[keyv] = best
        self.best_config = best
        if tuned_now:
            # benchmarking dirtied accumulation buffers; reset before the real run
            for name in self.reset_to_zero:
                bound[name].zero_()
        self.jitfn[grid](*args, **{**kwargs, **best.kwargs,
                                   "num_warps": best.num_warps,
                                   "num_stages": best.num_stages})


def autotune(configs, key, reset_to_zero=None):
    def deco(jitfn):
        return Autotuner(jitfn, configs, key, reset_to_zero)

    return deco


class Heuristics:
    def __init__(self, jitfn, values):
        self.jitfn = jitfn
        self.values = values
        self.__name__ = getattr(jitfn, "__name__", "kernel")
        self.param_names = jitfn.param_names

    def __getitem__(self, grid):
        def runner(*args, **kwargs):
            names = self.jitfn.param_names
            bound = dict(zip(names, args))
            bound.update(kwargs)
            for name, fn in self.values.items():
                kwargs[name] = fn(bound)
            return self.jitfn[grid](*args, **kwargs)

        return runner


def heuristics(values):
    def deco(jitfn):
        return Heuristics(jitfn, values)

    return deco
