#include <cuda_runtime.h>
#include <torch/extension.h>
// Baseline
template <typename scalar_t>
__global__ void reduce_sum_kernel0(const scalar_t *A, scalar_t *B, int n) {
  extern __shared__ scalar_t s_data[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = (idx < n) ? A[idx] : scalar_t(0);

  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
      s_data[tid] += s_data[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(B, s_data[0]);
}
template <typename scalar_t>
__global__ void reduce_sum_kernel1(const scalar_t *A, scalar_t *B, int N) {
  extern __shared__ __align__(sizeof(scalar_t)) char s_data_char[];
  scalar_t *s_data = reinterpret_cast<scalar_t *>(s_data_char);

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = (idx < N) ? A[idx] : scalar_t(0);

  __syncthreads();

  // 优化后的归约逻辑（避免线程束分化）
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(B, s_data[0]);
  }
}
// float4
template <typename scalar_t>
__global__ void reduce_sum_kernel2(const scalar_t *A, scalar_t *B, int N) {
  extern __shared__ __align__(sizeof(scalar_t)) char s_data_char[];
  scalar_t *s_data = reinterpret_cast<scalar_t *>(s_data_char);

  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  s_data[tid] = (idx < N) ? A[idx] : scalar_t(0);

  __syncthreads();

  // 优化后的归约逻辑（避免线程束分化）
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(B, s_data[0]);
  }
}
torch::Tensor reduce_sum(torch::Tensor &Input, int mod) {
  AT_ASSERTM(Input.device().is_cuda(), "Input must be a CUDA tensor");
  auto o = torch::zeros({1}, Input.options());
  auto n = Input.numel();
  dim3 block = 1024;
  dim3 grid = (n + block.x - 1) / block.x;
  switch (mod) {
  case 0:
    reduce_sum_kernel0<float><<<grid, block, 1024 * sizeof(float)>>>(
        Input.data_ptr<float>(), o.data_ptr<float>(), n);
    break;
  case 1:
    reduce_sum_kernel1<float><<<grid, block, 1024 * sizeof(float)>>>(
        Input.data_ptr<float>(), o.data_ptr<float>(), n);
    break;
  default:
    throw std::runtime_error("Invalid mode");
  }

  return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum", &reduce_sum, "reduce_sum");
}