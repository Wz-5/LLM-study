import math
import torch
from torch.autograd.function import once_differentiable


class FlashAttentionReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_q, block_k, causal):
        # q, k: [B, H, N, D]
        # v:    [B, H, N, Dv]
        n, d = q.shape[-2:]
        scale = 1.0 / math.sqrt(d)

        # 低精度输入使用 FP32 计算；保留 FP64 便于数值验证。
        dtype = (
            torch.float64
            if q.dtype == torch.float64
            else torch.float32
        )
        qf, kf, vf = [x.to(dtype) for x in (q, k, v)]

        out = torch.empty_like(vf)
        lse = torch.empty(
            q.shape[:-1], device=q.device, dtype=dtype
        )

        for i in range(0, n, block_q):
            ie = min(i + block_q, n)
            qi = qf[..., i:ie, :]

            # 每个 query 行分别维护 m、ell、acc。
            state_shape = qi.shape[:-1] + (1,)
            m = torch.full(
                state_shape, -torch.inf,
                device=q.device, dtype=dtype
            )
            ell = torch.zeros_like(m)
            acc = torch.zeros_like(vf[..., i:ie, :])

            # causal 模式下，当前 query 块不需要 ie 之后的 key。
            stop = ie if causal else n

            for j in range(0, stop, block_k):
                je = min(j + block_k, stop)
                kj = kf[..., j:je, :]
                vj = vf[..., j:je, :]

                # S_ij = Q_i K_j^T / sqrt(d)
                s = (qi @ kj.transpose(-2, -1)) * scale

                if causal:
                    rows = torch.arange(
                        i, ie, device=q.device
                    )[:, None]
                    cols = torch.arange(
                        j, je, device=q.device
                    )[None, :]
                    s = s.masked_fill(cols > rows, -torch.inf)

                # 在线 softmax 的最大值更新。
                m_new = torch.maximum(
                    m, s.amax(dim=-1, keepdim=True)
                )

                # 将旧状态换算到新的最大值基准。
                alpha = torch.exp(m - m_new)

                # 这里的 p 尚未归一化，对应推导中的 E_ij。
                p = torch.exp(s - m_new)

                ell = (
                    alpha * ell
                    + p.sum(dim=-1, keepdim=True)
                )
                acc = alpha * acc + p @ vj
                m = m_new

            out[..., i:ie, :] = acc / ell
            lse[..., i:ie] = (m + ell.log()).squeeze(-1)

        # 不保存任何完整 N×N 注意力矩阵。
        ctx.save_for_backward(qf, kf, vf, out, lse)
        ctx.config = (
            block_q, block_k, causal, scale, q.dtype
        )

        return out.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        block_q, block_k, causal, scale, input_dtype = (
            ctx.config
        )

        g = grad_out.to(q.dtype)

        # D_i = <G_i, O_i>
        delta = (g * out).sum(dim=-1, keepdim=True)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        n = q.shape[-2]

        for i in range(0, n, block_q):
            ie = min(i + block_q, n)
            qi = q[..., i:ie, :]
            gi = g[..., i:ie, :]
            stop = ie if causal else n

            for j in range(0, stop, block_k):
                je = min(j + block_k, stop)
                kj = k[..., j:je, :]
                vj = v[..., j:je, :]

                # 按块重算分数。
                s = (qi @ kj.transpose(-2, -1)) * scale

                if causal:
                    rows = torch.arange(
                        i, ie, device=q.device
                    )[:, None]
                    cols = torch.arange(
                        j, je, device=q.device
                    )[None, :]
                    s = s.masked_fill(cols > rows, -torch.inf)

                # 用前向保存的 logsumexp 恢复真实概率。
                p = torch.exp(s - lse[..., i:ie, None])

                # dP_ij = G_i V_j^T
                dp = gi @ vj.transpose(-2, -1)

                # dS_ij = P_ij * (dP_ij - D_i)
                ds = p * (dp - delta[..., i:ie, :])

                # 对所有相关块累积梯度。
                dq[..., i:ie, :] += (ds @ kj) * scale
                dk[..., j:je, :] += (
                    ds.transpose(-2, -1) @ qi
                ) * scale
                dv[..., j:je, :] += (
                    p.transpose(-2, -1) @ gi
                )

        return (
            dq.to(input_dtype),
            dk.to(input_dtype),
            dv.to(input_dtype),
            None,  # block_q
            None,  # block_k
            None,  # causal
        )


def flash_attention_reference(
    q, k, v, block_q=64, block_k=64, causal=False
):
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape
    assert q.shape[:3] == v.shape[:3]
    assert q.device == k.device == v.device
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    assert min(q.shape) > 0 and v.shape[-1] > 0
    assert block_q > 0 and block_k > 0

    return FlashAttentionReference.apply(
        q, k, v, block_q, block_k, causal
    )