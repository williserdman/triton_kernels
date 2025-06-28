"""
- basics
- testing
- benchmarking
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.jit # decorator tells triton to compile this function on the GPU
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr): # constant expr tells triton that this argument that this is static (known at compile time), any time we know we'll have these values do this
    # each program needs to know which chunk of data they have to process
    PID = tl.program_id(axis=0) # axis 0 refers to the grid # will either be 0, 1, 2, or 3 in this scenario
    # vector of length 256 and block size 64
    # pid 0 might be [0, 64)
    # pid 1 [64, 128)
    # etc

    block_start = PID * BLOCK_SIZE

    # instances this instance of kernel going to process
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # creates an array of length BLOCK_SIZE from [0, block_size)
    mask = offsets < n_elements # making sure that we're in the allowed range

    # load data from DRAM/VRAM/HBM to SRAM/on-chip-memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None) # pointer to all ### first elements (that this program is trying to process), PID 1 grabbing the next ### entires, etc.
    y = tl.load(y_ptr + offsets, mask=mask) # shape BLOCK_SIZE # default for other is None

    output = x + y # triton has these operations built in

    # write data back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask) # writes the actual data back to DRAM
    

def add(x, y):
    # pre allocate the output. the tensor for the answer must exist before we call the kernel
    output = torch.empty_like(x) # output is the EXACT SAME SHAPE so we can do this here

    # check that tensors on same device
    assert x.device == DEVICE and y.device == DEVICE
    # might need something like shape assertions too

    # total number of entries in that tensor
    n_elements = output.numel()

    # define out launch grid, number of programs that will run in parallel. when we call a kernel it runs the code 100-100k times in different SMs on the GPU. we have to define how many of those to call and how to differentiate them
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )# we want this to be a tuple # lambda is inline function... meta as in meta parameters
    # triton ceiling division, cdiv(m, n) = (m + (n - 1)) // n
    # grid is how many instances of kernel gonna be called up

    add_kernel[grid](
        x, 
        y, 
        output, 
        n_elements,
        BLOCK_SIZE = 1024
    ) # this calls the kernel... will write onto `output' (in place)

    return output # return the output tensor

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE): # atol is absolute tolerance b/c won't get exact same value compared to pytorch equivalance, rtos is relative tolerance.. idk what this is tbh

    # create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=device) # torch random normal distribution
    y = torch.randn(size, device=device)

    # run triton kernel & pytorch equivalent
    z_tri = add(x, y)
    z_ref = x + y # whatever pytorch has built in

    #compare them
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol) # check if they are close enough, tells what % of entries are INCORRECT
    print("pass")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"], # the x-axis of the graph, the different sizes of the input we want to test
        x_vals = (2**i for i in range(2, 20, 1)),
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="GB/s", # limiting factor normalls; either TFLOPS or GB/s typically
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider): # telling triton to benchmark different providers, either triton or torch
    # create input data
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95] # median and 5% percentile, 95% percentile

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == "triton":
        # see below, prim, min, max
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)# 3 is number of memory operations (load 2x + store)... 1e-9 converts to GBs from bytes.. then divide by seconds

    # NOTE: this is primary, max, and then min... however this is inconsistent with the above
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path=".", print_data=True)
