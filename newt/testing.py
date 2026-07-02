"""Benchmark helpers (CUDA-event timing, L2 flush) - mirrors triton.testing."""

import torch


def do_bench(fn, warmup=10, rep=50, quantile=0.5):
    """Median wall time of fn() in milliseconds, with L2 flushed between reps."""
    fn()  # trigger compilation
    torch.cuda.synchronize()
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        cache.zero_()  # flush L2
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start, end))
    return times[int(len(times) * quantile)]


def assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=""):
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        rel = diff / expected.abs().clamp_min(1e-6)
        raise AssertionError(
            f"{msg} mismatch: max abs err {diff.max().item():.3e}, "
            f"max rel err {rel.max().item():.3e}"
        )
