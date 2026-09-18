#include "gemm.cuh"

#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void tiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K) {

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    for(int phase = 0; phase < K / TILE_WIDTH; phase++) {

        A_s[row][col] =
            A[thready * K + (phase * TILE_WIDTH + col)];
        B_s[row][col] =
            B[(phase * TILE_WIDTH + row) * N + threadx];

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            sum += A_s[row][k] * B_s[k][col];
        }

        __syncthreads();
    }

    C[thready * N + threadx] = sum;
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
