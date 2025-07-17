"""
- backwards passes
- connecting backward pass (bwd) to pytorch graph
- reuse intermediate values from forward pass (fwd) to bwd
- locks and atomic operations
- benefits of two sequential kernels vs one single kernel
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def _layernorm_forward(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    stride_M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        sum_accumulator += x

    mean = tl.sum(sum_accumulator, axis=0) / N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        diff = tl.where(cols < N, x - mean, 0.0)
        acc += diff * diff

    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)  #  reciprocal standard dev.

    tl.store(mean_ptr + row, mean)  # important part of fwd pass
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask)

        x_normed = (x - mean) * rstd
        y = x_normed * w + b

        tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def _layernorm_backward_dLdx(
    x_ptr,
    dLdx_ptr,
    dLdy_ptr,
    w_ptr,
    dLdw_inter_ptr,
    dLdb_inter_ptr,
    mean_ptr,
    rstd_ptr,
    locks_ptr,
    stride,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)

    x_normed = tl.where(mask, (x - mean) * rstd, 0.0)
    dydx_normed = tl.where(mask, w * dLdy, 0.0)
    c1 = tl.sum(x_normed * dydx_normed, axis=0)
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normed * c1 + c2)) * rstd

    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    dLdw_cont = (dLdy * x_normed).to(w.dtype)
    dLdb_cont = dLdy.to(w.dtype)

    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE

    dLdw_inter_ptrs = dLdw_inter_ptr + lock_id * N + cols
    dLdb_inter_ptrs = dLdb_inter_ptr + lock_id * N + cols

    # calculations done beforehand so that we have to lock as little as possible
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass  # if it's 0 (unlocked), change it to 1 (lock it) and return 0 which will evaluate to False and let us leave while loop
        # if it's 1 (locked) we leave it as 1 and return 1 (evals to True) so we stay in loop

    count = tl.load(count_ptr)  # this count thing is "non-negligible??? maybe"
    if count == 0:  # no PID has used lock before
        tl.atomic_xchg(count_ptr, 1)

        tl.store(dLdw_inter_ptrs, dLdw_cont, mask=mask)
        tl.store(dLdb_inter_ptrs, dLdb_cont, mask=mask)

    else:
        dLdw_cont += tl.load(dLdw_inter_ptrs, mask=mask)
        dLdb_cont += tl.load(dLdb_inter_ptrs, mask=mask)

        tl.store(dLdw_inter_ptrs, dLdw_cont, mask=mask)
        tl.store(dLdb_inter_ptrs, dLdb_cont, mask=mask)

        tl.atomic_xchg(locks_ptr, 0)


@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_inter_ptr,
    dLdb_inter_ptr,
    dLdw_ptr,
    dLdb_ptr,
    GROUP_SIZE,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    PID = tl.program_id(0)
    col_offsets = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_offsets = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_offsets[:, None] < GROUP_SIZE) & (col_offsets[None, :] < N)
        offsets = row_offsets[:, None] * N + col_offsets[None, :]

        dLdw_acc += tl.load(dLdw_inter_ptr + offsets, mask=mask, other=0.0)
        dLdb_acc += tl.load(dLdb_inter_ptr + offsets, mask=mask, other=0.0)

    dLdw_chunk = tl.sum(dLdw_acc, axis=0)  # shape (BLOCK_SIZE_N)
    dLdb_chunk = tl.sum(dLdb_acc, axis=0)  # shape (BLOCK_SIZE_N)

    tl.store(dLdw_ptr + col_offsets, dLdw_chunk, mask=col_offsets < N)
    tl.store(dLdb_ptr + col_offsets, dLdb_chunk, mask=col_offsets < N)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):  # ctx is implied input
        M, N = x.reshape(-1, x.shape[-1]).shape

        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        MAX_FUSED_SIZE = (
            65536 // x.element_size()
        )  # can imrpove by switching to autotuning
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

        # if N > BLOCK_SIZE:
        #    raise RuntimeError("this layernorm doesn't support feature dim >= 64kb")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward[(M,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            x.stride(0),  # may need to be -2
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )  # every single row gets its own program

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dLdx = torch.empty_like(dLdy)  # (M, N)
        dLdw = torch.empty_like(
            w
        )  # (N), we will have M vectors of length N, we can't just use tl.store as that'll overwrite old data
        dLdb = torch.empty_like(
            b
        )  # (N)... ^ we could load, add, then store back, however, a bunch of PIDS are read/calc writing, so we have to use some locks

        GROUP_SIZE = 64
        if N <= 8192:
            GROUP_SIZE = 96
        if N <= 4096:
            GROUP_SIZE = 128
        if N <= 1024:
            GROUP_SIZE = 256

        dLdw_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_inter = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)

        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=x.device)

        _layernorm_backward_dLdx[(M,)](
            x,
            dLdx,
            dLdy,
            w,
            dLdw_inter,
            dLdb_inter,
            mean,
            rstd,
            locks,
            x.stride(0),
            N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_inter,
            dLdb_inter,
            dLdw,
            dLdb,
            min(GROUP_SIZE, M),
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )

        return (
            dLdx,
            None,
            dLdw,
            dLdb,
            None,
        )  # outputs have to correspont to inputs of forward pass, pytorch knows which is which based on ordering


layernorm = LayerNorm.apply


# step 1
def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    x.requires_grad_(True)

    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)

    y_tri = layernorm(x, (N,), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)
    print("passed fwd")

    dLdy = 0.1 * torch.randn_like(x)

    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)

    print("passed bwd")


test_layernorm_kernel(1151, 8192, torch.float16)
