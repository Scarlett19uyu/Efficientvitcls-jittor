import jittor as jt
from jittor import nn
import triton
import triton.language as tl

__all__ = ['TritonRMSNorm2dFunc']

@triton.jit
def _rms_norm_2d_fwd_fused(X, Y, W, B, Rrms, M, C, N, num_blocks, eps, BLOCK_SIZE: tl.constexpr):
    m_n = tl.program_id(0)
    (m, n) = ((m_n // num_blocks), (m_n % num_blocks))
    Y += ((m * C) * N)
    X += ((m * C) * N)
    cols = ((n * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE))
    mask = (cols < N)
    x_sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C):
        x = tl.load(((X + (off * N)) + cols), mask=mask, other=0.0).to(tl.float32)
        x_sum_square += (x * x)
    mean_square = (x_sum_square / C)
    rrms = (1 / tl.sqrt((mean_square + eps)))
    tl.store(((Rrms + (m * N)) + cols), rrms, mask=mask)
    for off in range(0, C):
        pos = ((off * N) + cols)
        w = tl.load((W + off))
        b = tl.load((B + off))
        x = tl.load((X + pos), mask=mask, other=0.0).to(tl.float32)
        x_hat = (x * rrms)
        y = ((x_hat * w) + b)
        tl.store((Y + pos), y, mask=mask)

@triton.jit
def _rms_norm_2d_bwd_dx_fused(DX, DY, DW, DB, X, W, B, Rrms, M, C, N, num_blocks, eps, 
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_C: tl.constexpr):
    m_n = tl.program_id(0)
    (m, n) = ((m_n // num_blocks), (m_n % num_blocks))
    X += ((m * C) * N)
    DY += ((m * C) * N)
    DX += ((m * C) * N)
    Rrms += (m * N)
    cols = ((n * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE))
    mask = (cols < N)
    DW = (DW + (m_n * C))
    DB = (DB + (m_n * C))
    rrms = tl.load((Rrms + cols), mask=mask, other=1)
    c1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C):
        pos = ((off * N) + cols)
        x = tl.load((X + pos), mask=mask, other=0).to(tl.float32)
        dy = tl.load((DY + pos), mask=mask, other=0).to(tl.float32)
        w = tl.load((W + off)).to(tl.float32)
        xhat = (x * rrms)
        wdy = (w * dy)
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 += (xhat * wdy)
        tl.store((DW + off), tl.sum((dy * xhat).to(w.dtype), axis=0))
        tl.store((DB + off), tl.sum(dy.to(w.dtype), axis=0))
    c1 /= C
    for off in range(0, C):
        pos = ((off * N) + cols)
        x = tl.load((X + pos), mask=mask, other=0).to(tl.float32)
        dy = tl.load((DY + pos), mask=mask, other=0).to(tl.float32)
        w = tl.load((W + off)).to(tl.float32)
        xhat = (x * rrms)
        wdy = (w * dy)
        dx = ((wdy - (xhat * c1)) * rrms)
        tl.store((DX + pos), dx, mask=mask)

class TritonRMSNorm2dFunc:
    """Jittor版本的Triton RMS归一化函数"""

    @staticmethod
    def execute(x, weight, bias, eps):
        y = jt.empty_like(x)
        x_arg = x.reshape(x.shape[0], x.shape[1], -1)  # [B,C,H,W] -> [B,C,H*W]
        M, C, N = x_arg.shape
        rrms = jt.empty((M, N), dtype='float32')
        
        BLOCK_SIZE = 256
        num_blocks = triton.cdiv(N, BLOCK_SIZE)
        num_warps = 8
        
        _rms_norm_2d_fwd_fused[(M * num_blocks,)](
            x_arg, y, weight, bias, rrms, 
            M, C, N, num_blocks, eps, 
            BLOCK_SIZE=BLOCK_SIZE, 
            num_warps=num_warps, 
            num_ctas=1
        )
        
        ctx = {
            'x': x,
            'weight': weight,
            'bias': bias,
            'rrms': rrms,
            'BLOCK_SIZE': BLOCK_SIZE,
            'num_blocks': num_blocks,
            'num_warps': num_warps,
            'eps': eps
        }
        return y, ctx

    @staticmethod
    def backward(ctx, dy):
        x, w, b, rrms = ctx['x'], ctx['weight'], ctx['bias'], ctx['rrms']
        num_blocks = ctx['num_blocks']
        
        x_arg = x.reshape(x.shape[0], x.shape[1], -1)  # 保持与前向相同的reshape
        M, C, N = x_arg.shape
        GROUP_SIZE_M = M * num_blocks
        
        _dw = jt.empty((GROUP_SIZE_M, C), dtype=x.dtype)
        _db = jt.empty((GROUP_SIZE_M, C), dtype=x.dtype)
        dw = jt.empty((C,), dtype=w.dtype)
        db = jt.empty((C,), dtype=w.dtype)
        dx = jt.empty_like(dy)
        
        _rms_norm_2d_bwd_dx_fused[(M * num_blocks,)](
            dx, dy, _dw, _db, x, w, b, rrms,
            M, C, N, num_blocks, ctx['eps'],
            BLOCK_SIZE=ctx['BLOCK_SIZE'],
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_C=triton.next_power_of_2(C),
            num_warps=ctx['num_warps']
        )
        
        dw = _dw.sum(dim=0)
        db = _db.sum(dim=0)
        return dx, dw, db, None