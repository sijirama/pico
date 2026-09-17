#include "gemm.cuh"

#include <cuda_runtime.h>

__global__ void double_buffered_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K
) {
    // cp.async degeneracy begins here
}


void cuda_gemm_double_buffered(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K
) {
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    double_buffered_kernel<<<grid, block>>>(
        A, B, C,
        M, N, K
    );
}
