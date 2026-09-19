#include "gemm.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void
double_buffer(const float *A, const float *B, float *C, int M, int N, int K) {

    int row = threadIdx.y;
    int col = threadIdx.x;

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.z;

    const float *A_current = A + batch_id * M * K;
    const float *B_current = B + batch_id * K * N;
    float *C_current = C + batch_id * M * N;

    __shared__ float A_s[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[2][TILE_WIDTH][TILE_WIDTH];

    // prologue

    __pipeline_memcpy_async(
        &A_s[0][row][col],
        &A_current[thready * K + (0 * TILE_WIDTH + col)],
        sizeof(float));

    __pipeline_memcpy_async(
        &B_s[0][row][col],
        &B_current[(0 * TILE_WIDTH + row) * N + threadx],
        sizeof(float));

    __pipeline_commit();

    float sum = 0.0f;
    for(int phase = 0; phase < K / TILE_WIDTH; phase++) {

        int nextPhase = phase + 1;
        int currentStage = phase % 2;
        int nextStage = nextPhase % 2;

        if(nextPhase < K / TILE_WIDTH) {

            __pipeline_memcpy_async(
                &A_s[nextStage][row][col],
                &A_current[thready * K + (nextPhase * TILE_WIDTH + col)],
                sizeof(float));

            __pipeline_memcpy_async(
                &B_s[nextStage][row][col],
                &B_current[(nextPhase * TILE_WIDTH + row) * N + threadx],
                sizeof(float));

            __pipeline_commit();
        }

        // "allow N newest groups to remain unfinished"
        if(nextPhase < K / TILE_WIDTH) {
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            sum += A_s[currentStage][row][k] * B_s[currentStage][k][col];
        }

        __syncthreads();
    }

    C_current[thready * N + threadx] = sum;
}

void cuda_gemm_double_buffered(
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
    double_buffer<<<grid, block>>>(A, B, C, M, N, K);
}
