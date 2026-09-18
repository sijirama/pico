#include "gemm.cuh"

#include <cuda_runtime.h>

__global__ void naive(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K) {

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thready >= M || threadx >= N)
        return;

    float sum = 0.0f;
    for(int k = 0; k < K; k++) {
        sum += A[thready * K + k] * B[k * N + threadx];
    }
    C[thready * N + threadx] = sum;
}

void cuda_gemm_naive(
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

    naive<<<grid, block>>>(A, B, C, M, N, K);
}
