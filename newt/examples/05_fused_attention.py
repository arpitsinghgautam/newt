"""Fused attention (flash-attention-style forward, non-causal, single head).

Demonstrates the full newt op set working together: tensor-core dots,
online-softmax rescaling (the accumulator moves between fragment and register
layouts), trans, masked loads, and loop-carried block state.
"""

import torch

import newt
import newt.language as nl


@newt.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, M, N, scale,
                     sqm, sqd, skn, skd, svn, svd, som, sod,
                     BLOCK_M: nl.constexpr, BLOCK_N: nl.constexpr, D: nl.constexpr):
    pid = nl.program_id(0)
    offs_m = pid * BLOCK_M + nl.arange(0, BLOCK_M)
    offs_d = nl.arange(0, D)
    qmask = offs_m[:, None] < M
    q = nl.load(q_ptr + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                mask=qmask, other=0.0)                                # [BM, D]

    m_i = nl.full((BLOCK_M,), float("-inf"), dtype=nl.float32)       # row maxes
    l_i = nl.zeros((BLOCK_M,), dtype=nl.float32)                     # row sums
    acc = nl.zeros((BLOCK_M, D), dtype=nl.float32)                   # output acc

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + nl.arange(0, BLOCK_N)
        nmask = offs_n[:, None] < N
        k = nl.load(k_ptr + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=nmask, other=0.0)                            # [BN, D]
        qk = nl.dot(q, nl.trans(k)) * scale                          # [BM, BN]
        qk = nl.where(offs_n[None, :] < N, qk, float("-inf"))
        m_new = nl.maximum(m_i, nl.max(qk, axis=1))
        alpha = nl.exp(m_i - m_new)
        p = nl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + nl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = nl.load(v_ptr + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=nmask, other=0.0)                            # [BN, D]
        acc = nl.dot(p.to(nl.float16), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    nl.store(o_ptr + offs_m[:, None] * som + offs_d[None, :] * sod,
             acc, mask=qmask)


def attention(q, k, v, BLOCK_M=64, BLOCK_N=64, num_warps=4):
    M, D = q.shape
    N = k.shape[0]
    o = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)
    grid = (newt.cdiv(M, BLOCK_M),)
    attention_kernel[grid](
        q, k, v, o, M, N, scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), o.stride(0), o.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, num_warps=num_warps,
    )
    return o


def main():
    torch.manual_seed(0)
    M = N = 1024
    D = 64
    q = torch.randn(M, D, device="cuda", dtype=torch.float16)
    k = torch.randn(N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(N, D, device="cuda", dtype=torch.float16)

    out = attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None]
    )[0, 0]
    err = (out - ref).abs().max().item()
    print(f"fused attention {M}x{N} D={D}: max err vs SDPA = {err:.4e}")
    assert err < 2e-2, "attention mismatch"

    t_newt = newt.testing.do_bench(lambda: attention(q, k, v))
    t_sdpa = newt.testing.do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None]))
    print(f"newt {t_newt:.3f} ms | torch SDPA {t_sdpa:.3f} ms "
          f"(GPU shared - numbers indicative only)")


if __name__ == "__main__":
    main()
