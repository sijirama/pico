#include "gemm.cuh"

#include <cuda_runtime.h>

__global__ void
naive(const float *A, const float *B, float *C, int M, int N, int K) {

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int current_matrix = blockIdx.z;

    if(thready >= M || threadx >= N)
        return;

    const float *A_current = A + current_matrix * M * K;
    const float *B_current = B + current_matrix * K * N;
    float *C_current = C + current_matrix * M * N;

    float sum = 0.0f;
    for(int k = 0; k < K; k++) {
        sum += A_current[thready * K + k] * B_current[k * N + threadx];
    }
    C_current[thready * N + threadx] = sum;
}

void cuda_gemm_naive(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_count) {

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);

    naive<<<grid, block>>>(A, B, C, M, N, K);
}
