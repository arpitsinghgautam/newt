"""newt vs triton(-windows) vs torch benchmark suite.

Usage (from the newt/ directory):
    python benchmarks/bench.py                 # full suite
    python benchmarks/bench.py --quick         # small sizes, low reps
    python benchmarks/bench.py --kernel matmul --save

Identical kernels (modulo nl<->tl) run for newt and triton; both frameworks
get the same small config sweep per size and report their best - the standard
way Triton's own benchmarks are run.
"""

import argparse
import os
import subprocess
import time

import torch

import newt
import newt.language as nl
from newt.testing import do_bench

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


def gpu_busy_warning():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        util = int(out.splitlines()[0])
        if util > 20:
            print("=" * 72)
            print(f"WARNING: GPU is {util}% utilized by other processes.")
            print("Benchmark numbers will be unreliable. Results are indicative only.")
            print("=" * 72)
            return True
    except Exception:
        pass
    return False


# ===========================================================================
# kernels
# ===========================================================================

@newt.jit
def n_add(x_ptr, y_ptr, o_ptr, n, BLOCK: nl.constexpr):
    offs = nl.program_id(0) * BLOCK + nl.arange(0, BLOCK)
    mask = offs < n
    x = nl.load(x_ptr + offs, mask=mask)
    y = nl.load(y_ptr + offs, mask=mask)
    nl.store(o_ptr + offs, x + y, mask=mask)


@newt.jit
def n_softmax(x_ptr, o_ptr, N, stride, BLOCK_N: nl.constexpr):
    row = nl.program_id(0)
    cols = nl.arange(0, BLOCK_N)
    mask = cols < N
    x = nl.load(x_ptr + row * stride + cols, mask=mask, other=float("-inf"))
    x = x - nl.max(x, axis=0)
    num = nl.exp(x)
    den = nl.sum(num, axis=0)
    nl.store(o_ptr + row * stride + cols, num / den, mask=mask)


@newt.jit
def n_layernorm(x_ptr, w_ptr, b_ptr, o_ptr, N, stride, eps, BLOCK_N: nl.constexpr):
    row = nl.program_id(0)
    cols = nl.arange(0, BLOCK_N)
    mask = cols < N
    x = nl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    mean = nl.sum(x, axis=0) / N
    diff = nl.where(mask, x - mean, 0.0)
    var = nl.sum(diff * diff, axis=0) / N
    rstd = nl.rsqrt(var + eps)
    w = nl.load(w_ptr + cols, mask=mask)
    b = nl.load(b_ptr + cols, mask=mask)
    nl.store(o_ptr + row * stride + cols, (x - mean) * rstd * w + b, mask=mask)


@newt.jit
def n_matmul(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BLOCK_M: nl.constexpr, BLOCK_N: nl.constexpr, BLOCK_K: nl.constexpr):
    pid_m = nl.program_id(0)
    pid_n = nl.program_id(1)
    offs_m = pid_m * BLOCK_M + nl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + nl.arange(0, BLOCK_N)
    offs_k = nl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = nl.zeros((BLOCK_M, BLOCK_N), dtype=nl.float32)
    for k in range(K // BLOCK_K):
        a = nl.load(a_ptrs)
        b = nl.load(b_ptrs)
        acc = nl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    nl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


if HAS_TRITON:
    @triton.jit
    def t_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, x + y, mask=mask)

    @triton.jit
    def t_softmax(x_ptr, o_ptr, N, stride, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=float("-inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(o_ptr + row * stride + cols, num / den, mask=mask)

    @triton.jit
    def t_layernorm(x_ptr, w_ptr, b_ptr, o_ptr, N, stride, eps, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / N
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        tl.store(o_ptr + row * stride + cols, (x - mean) * rstd * w + b, mask=mask)

    @triton.jit
    def t_matmul(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
        b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K // BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * sak
            b_ptrs += BLOCK_K * sbk
        c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


# ===========================================================================
# benchmarks
# ===========================================================================

def best_time(fn_for_cfg, configs, warmup, rep):
    best = None
    for cfg in configs:
        try:
            t = do_bench(fn_for_cfg(cfg), warmup=warmup, rep=rep)
        except Exception:
            continue
        if best is None or t < best:
            best = t
    return best


def bench_add(sizes, warmup, rep):
    rows = []
    configs = [(1024, 4), (4096, 4), (4096, 8), (16384, 8)]
    for n in sizes:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        o = torch.empty_like(x)
        gb = 3 * n * 4 / 1e9
        t_torch = do_bench(lambda: torch.add(x, y, out=o), warmup=warmup, rep=rep)
        t_newt = best_time(
            lambda c: lambda: n_add[(newt.cdiv(n, c[0]),)](x, y, o, n, BLOCK=c[0],
                                                           num_warps=c[1]),
            configs, warmup, rep)
        t_tri = best_time(
            lambda c: lambda: t_add[(newt.cdiv(n, c[0]),)](x, y, o, n, BLOCK=c[0],
                                                           num_warps=c[1]),
            configs, warmup, rep) if HAS_TRITON else None
        rows.append((f"{n // (1 << 20)}M", gb, t_torch, t_newt, t_tri))
    return "vector add fp32 (GB/s)", rows


def bench_softmax(Ns, warmup, rep, M=4096):
    rows = []
    for N in Ns:
        x = torch.randn(M, N, device="cuda")
        o = torch.empty_like(x)
        gb = 2 * M * N * 4 / 1e9
        BN = newt.next_power_of_2(N)
        configs = [(4,), (8,), (16,)]
        t_torch = do_bench(lambda: torch.softmax(x, -1, out=o), warmup=warmup, rep=rep)
        t_newt = best_time(
            lambda c: lambda: n_softmax[(M,)](x, o, N, x.stride(0), BLOCK_N=BN,
                                              num_warps=c[0]),
            configs, warmup, rep)
        t_tri = best_time(
            lambda c: lambda: t_softmax[(M,)](x, o, N, x.stride(0), BLOCK_N=BN,
                                              num_warps=c[0]),
            configs, warmup, rep) if HAS_TRITON else None
        rows.append((f"{M}x{N}", gb, t_torch, t_newt, t_tri))
    return "fused softmax fp32 (GB/s)", rows


def bench_layernorm(Ns, warmup, rep, M=4096):
    rows = []
    for N in Ns:
        x = torch.randn(M, N, device="cuda")
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        o = torch.empty_like(x)
        gb = 2 * M * N * 4 / 1e9
        BN = newt.next_power_of_2(N)
        configs = [(4,), (8,), (16,)]
        t_torch = do_bench(
            lambda: torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5),
            warmup=warmup, rep=rep)
        t_newt = best_time(
            lambda c: lambda: n_layernorm[(M,)](x, w, b, o, N, x.stride(0), 1e-5,
                                                BLOCK_N=BN, num_warps=c[0]),
            configs, warmup, rep)
        t_tri = best_time(
            lambda c: lambda: t_layernorm[(M,)](x, w, b, o, N, x.stride(0), 1e-5,
                                                BLOCK_N=BN, num_warps=c[0]),
            configs, warmup, rep) if HAS_TRITON else None
        rows.append((f"{M}x{N}", gb, t_torch, t_newt, t_tri))
    return "layernorm fwd fp32 (GB/s)", rows


MM_CONFIGS = [
    (128, 128, 32, 8, 2), (128, 128, 32, 8, 3), (128, 128, 32, 8, 4),
    (128, 128, 64, 8, 2), (128, 128, 64, 8, 3),
    (128, 256, 32, 8, 3), (64, 128, 64, 8, 3), (64, 128, 64, 8, 4),
    (64, 128, 64, 4, 3), (128, 64, 64, 8, 3), (64, 64, 64, 4, 3),
    (128, 128, 32, 4, 3),
]


def bench_matmul(sizes, warmup, rep, dtype=torch.float16):
    rows = []
    for size in sizes:
        M = N = K = size
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = torch.empty(M, N, device="cuda", dtype=dtype)
        tf = 2 * M * N * K / 1e12

        def newt_fn(cfg):
            BM, BN, BK, W, S = cfg
            grid = (newt.cdiv(M, BM), newt.cdiv(N, BN))
            return lambda: n_matmul[grid](
                a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                num_warps=W, num_stages=S)

        def tri_fn(cfg):
            BM, BN, BK, W, S = cfg
            grid = (newt.cdiv(M, BM), newt.cdiv(N, BN))
            return lambda: t_matmul[grid](
                a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                num_warps=W, num_stages=S)

        t_torch = do_bench(lambda: torch.matmul(a, b, out=c), warmup=warmup, rep=rep)
        t_newt = best_time(newt_fn, MM_CONFIGS, warmup, rep)
        t_tri = best_time(tri_fn, MM_CONFIGS, warmup, rep) if HAS_TRITON else None
        rows.append((f"{size}^3", tf, t_torch, t_newt, t_tri))
    name = f"matmul {str(dtype).split('.')[-1]} (TFLOP/s)"
    return name, rows


def bench_matmul_tf32(sizes, warmup, rep):
    torch.backends.cuda.matmul.allow_tf32 = True
    name, rows = bench_matmul(sizes, warmup, rep, dtype=torch.float32)
    return "matmul fp32/tf32 (TFLOP/s)", rows


# ===========================================================================
# reporting
# ===========================================================================

def fmt_table(title, rows, unit_is_tflops):
    lines = [f"### {title}", ""]
    hdr = "| shape | torch | newt | triton |" if HAS_TRITON else "| shape | torch | newt |"
    sep = "|---" * (4 if HAS_TRITON else 3) + "|"
    lines += [hdr, sep]
    for shape, work, t_torch, t_newt, t_tri in rows:
        def cell(t):
            if t is None:
                return "n/a"
            v = work / t * 1e3
            return f"{v:.1f}" if unit_is_tflops else f"{v:.0f}"

        row = f"| {shape} | {cell(t_torch)} | {cell(t_newt)} |"
        if HAS_TRITON:
            row += f" {cell(t_tri)} |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--kernel", default=None,
                    choices=[None, "add", "softmax", "layernorm", "matmul", "matmul_tf32"])
    ap.add_argument("--cooldown", type=int, default=0,
                    help="seconds to idle between suites (laptops throttle; "
                         "each suite should start from a similar thermal state)")
    args = ap.parse_args()

    busy = gpu_busy_warning()
    warmup, rep = (3, 10) if args.quick else (10, 50)
    if args.quick:
        add_sizes, sm_Ns, ln_Ns, mm_sizes = [1 << 22], [1024], [1024], [1024]
    else:
        add_sizes = [1 << 20, 1 << 24, 1 << 26, 1 << 27]
        sm_Ns = [1024, 4096, 8192, 16384]
        ln_Ns = [1024, 2048, 4096, 8192]
        mm_sizes = [1024, 2048, 4096, 8192]

    suites = {
        "add": lambda: bench_add(add_sizes, warmup, rep),
        "softmax": lambda: bench_softmax(sm_Ns, warmup, rep),
        "layernorm": lambda: bench_layernorm(ln_Ns, warmup, rep),
        "matmul": lambda: bench_matmul(mm_sizes, warmup, rep),
        "matmul_tf32": lambda: bench_matmul_tf32(mm_sizes, warmup, rep),
    }
    if args.kernel:
        suites = {args.kernel: suites[args.kernel]}

    dev = torch.cuda.get_device_name()
    out = [f"# newt benchmarks - {dev}", ""]
    if busy:
        out.append("> WARNING: GPU was shared with other processes during this run.\n")
    if not HAS_TRITON:
        out.append("> triton not installed; skipping triton columns.\n")
    for i, (name, fn) in enumerate(suites.items()):
        if i and args.cooldown:
            print(f"cooling down {args.cooldown}s...", flush=True)
            time.sleep(args.cooldown)
        print(f"running {name}...", flush=True)
        title, rows = fn()
        table = fmt_table(title, rows, "TFLOP" in title)
        print(table)
        out.append(table)

    if args.save:
        path = os.path.join(os.path.dirname(__file__), "results.md")
        with open(path, "w") as f:
            f.write("\n".join(out))
        print(f"saved to {path}")


if __name__ == "__main__":
    main()
