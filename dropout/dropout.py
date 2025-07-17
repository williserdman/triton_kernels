"""
- parallel pseudo random number generation in SRAM
-
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def _seeded_dropout_kernel(x_ptr, y_ptr, n_el, p, seed, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)

    offsets = PID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_el

    x = tl.load(x_ptr + offsets, mask=mask)

    random = tl.rand(
        seed, offsets
    )  # uniform distribution 0 to 1 array of length block size between 0 and 1

    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)  # 1-p is dropout specific math

    tl.store(y_ptr + offsets, output, mask=mask)


def seeded_dropout(x, prob, seed):
    output = torch.empty_like(x)

    assert x.is_contiguous()

    n_el = x.numel()
    grid = lambda meta: (triton.cdiv(n_el, meta["BLOCK_SIZE"]),)

    _seeded_dropout_kernel[grid](x, output, n_el, prob, seed, BLOCK_SIZE=1024)

    return output


x = torch.randn(size=(8,), device=DEVICE)
output1 = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=321)

print(x, output1, output2, output3)
