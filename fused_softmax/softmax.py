"""
- reduce memory read/write by fusing a kernel
- how to get GPU specifications
- more details on GPU architecture (better understanding)
- how to define meta parameters using heuristics and GPU specific attributes
- more about masking and how to choose the value of extra masked out entires
"""

import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

# step 1
def naive_softmax(x):
    # assume input is of size (M, N)
    # "safe softmax", don't wan't some very large values that get the exponential and become too large for float32/16
    # reads MN elements and write M elements
    x_max = x.max(dim=1)[0] # shape (M)

    # reads MN + M elements, subtraction is MN flops, write MN elements
    z = x - x_max[:, None] # shape (M, N) - shape(M, 1) == shape(M, N)

    # reading MN elements and writing MN elements
    numerator = torch.exp(x)

    # read MN elements, MN flops, write M elements
    denominator = numerator.sum(1) # shape (M, N) -> shape (M)

    # read MN + M elements, division MN flops, write MN elements
    out = numerator / denominator[:, None] # (M, N) / (M, 1) = shape (M, N)

    # ** naive version not fused even though pytorch would do some optimization on its own

    # we did this in total with 8MN + 4M memory operations... memory operations are super slow (slower than flops!)
    return out

# we want to read x one time and then write the output one time.

# step 4
@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr, 
    input_row_stride, output_row_stride,
    m_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr # we didn't pass these when calling the kernel, in this case we passed them in during the warmup
):
    # shape (M, N)
    # BLOCK_SIZE = next power of 2 larger than N
    # entire row into SRAM at a time
    # each PID going to start in a different ROW

    PID = tl.program_id(0)

    # if row_step smaller than number of rows it would have to jump down multiple rows
    row_step = tl.num_programs(0)
        # if 4 programs then row_step = 4
        # if n_rows = 6
        # PID 0 would get row 0
        # PID 1 row 1
        # PID 2 row 2
        # PID 3 row 3
        # PID 0 += row_step so now gets row 4
        # PID 1 += row_step == 5

    for row_idx in tl.range(PID, m_rows, row_step, num_stages=num_stages):
        # GPU can try to do lines of code that we wrote sequentially in parallel if we wrote them at once..
        # maybe different iterations of for loop at the same time. just know that triton is parallelizing under the hood
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # this points to a single entry in input tensor
        
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # now this points to every column in the tensor
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float("-inf")) # other is specific choice that fits well with softmax, shape BLOCKSIZE.. roughly (n_cols)
        # this will be the only time we read from memory

        # called fusing when doing multiple oprations within one kernel
        row_minus_max = row - tl.max(row, axis=0) # (BLOCK_SIZE) - (1) -> (BLOCK_SIZE)
        numerator = tl.exp(row_minus_max) # shape (BLOCK_SIZE)
        denominator = tl.sum(numerator, axis=0) # shape (1)
        softmax_output = numerator / denominator # (BLOCKSIZE) / (1) -> (BLOCK_SIZE)
        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)



# step 3
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] # "streaming multiprocessor"
NUM_REGS = properties["max_num_regs"] # registers fastest memory in the GPU, closer in than SRAM. each SM has limited number of registers. programs share registers
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"] # warp is smallest possible group of GPU cores, 32 cores per warp means that warp can work on 32 entires of a tensor simultaneously

def softmax(x):
    """ prep and call kernel, these don't connect to pytorch backprop graph"""
    assert x.ndim == 2
    assert x.is_contiguous()

    m_rows, n_cols = x.shape
    # assume every row of x fits within SRAM, don't want to have to calculate softmax across SMs

    BLOCK_SIZE = triton.next_power_of_2(n_cols) # best practice to do a power of 2

    # how many warps to give to a single program
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16

    # pipelining... even within an SM the GPU can do memory reads while it's doing FLOPs
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2 # "lines" that can be done simultaneously

    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(
        x, y,
        x.stride(0), y.stride(0),
        m_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs # will tell us registers needed per program
    sram_needed_per_program = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps) # NUM_REGS is total per SM
        # NUM_REGS = 65536
        # each program might use 
            # n_regs_per_program = 32
            # WARP_SIZE = 32
            # num_warps = 8
        # so each program needs (n_regs_per_program * WARP_SIZE * num_warps) registers total
        # 65536 // (32 * 32 * 8) = 8 programs per SM

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    programs_per_sm = min(reg_occupancy, sram_occupancy)

    num_programs = min(NUM_SM * programs_per_sm, m_rows)

    grid = (num_programs, 1, 1)

    kernel[grid](
        x, y,
        x.stride(0), y.stride(0), # relates to how x and y are stored in memory
            # x.stride() returns tuple with how many steps you have to take forward to get to next entry along that dimension
                # x is (M, N)
                # x.stride would be (N, 1)
                # x.stride(0) would be (N)
                # x.stride(1) would be 1
            # if you want to move one forward along N dimension (inner) you move forward one place,
            # if you want to move one forward along M dimension you move N forward
            # sometimes strides get changed to avoid read/write in reshape and stuff breaking the contiguous tensor 
        m_rows, n_cols,
        BLOCK_SIZE,
        num_stages
    )

    return y

# step 2
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) is tuple and len(size) == 2

    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=DEVICE)
    
    z_tri = softmax(x)
    z_ref = torch.softmax(x, axis=1)

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("pass")


# step 5
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ["N"],
        x_vals = [128 * i for i in range(2, 100)],
        line_arg = "provider",
        line_vals = ["triton", "torch"],
        line_names = ["Triton", "Torch"],
        styles = [("blue", "-"), ("orange", "-")],
        ylabel = "GB/s",
        plot_name = "softmax-performance",
        args={"M": 4096}
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    # maybe useful when other things are happening on your GPU, will track specifically our benchmark
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3) # 2 is number of memory operations (one load one store)
    return gbps(ms)


if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel(size=(1823, 781))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=".", print_data=True)