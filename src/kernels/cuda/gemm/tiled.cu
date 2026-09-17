#include "gemm.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

__global__ void tiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K) {

    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int thready = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for(int k = 0; k < K; k++) {
        sum += A[threadx * K + k] * B[k * N + thready];
    }

    C[threadx * N + thready] = sum;
}

void cuda_gemm_tiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K) {

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);
    tiled<<<grid, block>>>(A, B, C, M, N, K);
}
