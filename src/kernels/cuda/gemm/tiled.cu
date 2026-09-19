#include "gemm.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void
tiled(const float *A, const float *B, float *C, int M, int N, int K) {

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.z;

    int row = threadIdx.y;
    int col = threadIdx.x;

    const float *A_current = A + batch_id * M * K;
    const float *B_current = B + batch_id * K * N;
    float *C_current = C + batch_id * M * N;

    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    for(int phase = 0; phase < K / TILE_WIDTH; phase++) {

        A_s[row][col] = A_current[thready * K + (phase * TILE_WIDTH + col)];
        B_s[row][col] = B_current[(phase * TILE_WIDTH + row) * N + threadx];

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            sum += A_s[row][k] * B_s[k][col];
        }

        __syncthreads();
    }

    C_current[thready * N + threadx] = sum;
}

void cuda_gemm_tiled(
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
    tiled<<<grid, block>>>(A, B, C, M, N, K);
}
