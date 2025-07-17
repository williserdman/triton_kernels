"""
- automatic performance tuning
- PID reordering for improved SRAM sharing between PIDs
- multi-dimensional pointer arithmetic
- data types - high precision accumulation
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

# import os
# os.environ["TRITON_INTERPRET"] = "1" # makes triton simulate with numpy, you can use print statements... maybe works?

# step 3
# BLOCK_SIZE_M, BLOCK_SIZE_N
autotune_configs = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE": 8,
        },  # for stuff we defined the name of
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=5,
        num_warps=2,
    ),
    # Good config for fp8 inputs.
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
]


# to trigger autotuning
@triton.autotune(configs=autotune_configs, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_a_M,
    strike_a_K,
    stride_b_K,
    stride_b_N,
    stride_c_M,
    stride_c_N,
    # meta parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    define each instance of kernel by the block of c it's going to compute

    imagine M = N = K = 8
    BLOCK_SIZE_M/K/N = 2
    [0   1   2   3]
    [4   5   6   7]
    [8   9  10  11]
    [12 13  14  15]

    when we want to compute PID 0, it's a 2x2 chunk of our 8x8 matrix

    if we want to compute that chunk of C then we iterate through 4 chunks of a (row) and 4 chunks of b (column)

    a11 @ b11 + a12 @ b21 + a13 @ b31 + a14 @ b 41 = final value for 0th chunk

    multiple PIDs on a single SM... if we load exact same data we can reduce SRAM usage, this means being smart about what PIDs are doing.

    we want successing PIDs to share, (0, 1) or (0, 4) NOT (0, 15) which shares not much

    PID 0, 1, 2, 3: entire first row of a is duplicate! (good) then we load columns 1, 2, 3, 4 of b for a total of 5 columns..

    if we instead work with 0, 1, 4, 5... we only load 4 rows/columns (EVEN BETTER!)

    so if we can create a different ordering of our PIDs to tage advantage of this we can do groupwise ordering.

    "group major ordering scheme"
    """

    PID = tl.program_id(axis=0)

    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N

    group_id = PID // num_PID_in_group

    first_PID_in_group_along_M = group_id * GROUP_SIZE

    # just in case group is over the edge of the tensor
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE)

    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = (
        offsets_M[:, None] * stride_a_M + offsets_K[None, :] * strike_a_K
    )  # xxx[None, : ] same as xxx.expand_dims(0)
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    acc = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )  # tiny chunk of c # all input was in float16

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = (
            offsets_K < K - k * BLOCK_SIZE_K
        )  # if current starting index greater than dimension
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)

        acc = tl.dot(a, b, acc)

        a_ptr += BLOCK_SIZE_K * strike_a_K
        b_ptr += BLOCK_SIZE_K * stride_b_K

    c = acc.to(tl.float16)

    c_offsets = offsets_M[:, None] * stride_c_M + offsets_N[None, :] * stride_c_N
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, c, mask=c_mask)


# step 2
def matmul(a, b):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]

    (M, K), (_, N) = a.shape, b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.c_div(N, meta["BLOCK_SIZE_N"])
    )  # (number of blocks, )

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("orange", "-"), ("blue", "-")],
        ylabel="GB/s",  # we think TFLOPS will be the slowdown
        plot_name="matmul-performance-report",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    quantiles = [0.5, 0.05, 0.95]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles)

    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


# step 1
def test_matmul_kernel(
    size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE
):  # when choosing tolerances, doing a lot of FLOPS in kernel will deviate from PyTorch values (accumulation lots of FLOPS).. because of precision stuff we'll be going in a different order and therefore not really being super identical
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2

    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)

    torch.testing.assert_close(c_tri, c_ref, atol, rtol)
    print("pass")


if __name__ == "__main__":
    test_matmul_kernel(size=(512, 512))

    import sys

    if len(sys.argv) > 1 and sys.argv == "--benchmark":
        benchmark.run(save_path=".", print_data=False)
