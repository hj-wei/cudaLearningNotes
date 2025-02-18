import torch

import triton
import triton.language as tl
import torch_add_test


# @triton.jit
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    o = x + y
    tl.store(o_ptr + offsets, o, mask)


def add(x, y):
    o = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and o.is_cuda
    n_elements = o.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, o, n_elements, BLOCK_SIZE=1024)
    return o


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output_torch = x + y
output_triton = add(x, y)
output_cuda = torch_add_test.vector_add(x, y)
torch.cuda.synchronize()
print(output_torch)
print(output_triton)
print(output_cuda)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_cuda))}"
)
