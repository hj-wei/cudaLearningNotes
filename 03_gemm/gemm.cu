#include <torch/extension.h>
#include <cuda_runtime.h>
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// Normal gemm

__global__ void gemm1(float *A, float *B, float *O, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N)
    {
        float sum = 0;
        for (int i = 0; i < K; ++i)
        {
            sum += A[x * K + i] * B[i * N + y];
        }
    }
}
template <const int BLOCK_SIZE>
__global__ void gemm2(const float *A, const float *B, float *O, int M, int N, int K)
{
    __shard__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shard__ float Bs[BLOCK_SIZE * BLOCK_SIZE];
    int idx = threadIdx.x;
    int tx = idx % BLOCK_SIZE;
    int ty = idx / BLOCK_SIZE;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    A = &A[by * BLOCK_SIZE * K];
    B = &B[bx * BLOCK_SIZE];
    O = &O[by * BLOCK_SIZE * N + bx * BLOCK_SIZE];
    float sum = 0;
    for (int k = 0; k < K; k += BLOCK_SIZE)
    {
        As[ty * BLOCK_SIZE + tx] = A[ty * K + tx];
        Bs[ty * BLOCK_SIZE + tx] = B[ty * N + tx];
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            sum += As[ty * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tx];
        }
        _syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }
    O[ty * N + tx] = sum;
}
torch::Tensor
gemm_plugin(torch::Tensor A, torch::Tensor B, int mod)
{
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto O = torch::zeros({M, N}, A.options());
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();
    auto O_data = O.data_ptr<float>();
    switch (mod)
    {
    case 1:
        dim3 block(32, 32);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        gemm1<<<grid, block>>>(A_data, B_data, O_data, M, N, K);
        break;
    case 2:
        dim3 block(1024);
        dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        gemm2<32><<<grid, block>>>(A_data, B_data, O_data, M, N, K);
        break;
    default:
        break;
    }
    return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_plugin", &gemm_plugin, "gemm_plugin");
}