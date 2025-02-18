#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_cuda(const float *a, const float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

torch::Tensor add_cuda_torch(const torch::Tensor &x, const torch::Tensor &y) {
  auto z = torch::empty_like(x);
  const int N = x.numel();
  const dim3 block(1024);
  const dim3 grid((N + block.x - 1) / block.x);
  add_cuda<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                            z.data_ptr<float>(), N);

  return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &add_cuda_torch, "CUDA Vector Add");
}