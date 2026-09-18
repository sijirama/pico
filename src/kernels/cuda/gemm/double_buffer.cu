#include "gemm.cuh"

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

namespace cg = cooperative_groups;

__global__ void double_buffer(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K) {

    auto block = cg::this_thread_block();

    __shared__ cuda::
        pipeline_shared_state<cuda::thread_scope_block, 2>
            shared_state;

    // Create the pipeline handler
    cuda::pipeline pipeline =
        cuda::make_pipeline(block, &shared_state);

    int row = threadIdx.y;
    int col = threadIdx.x;

    int thready = blockIdx.y * blockDim.y + threadIdx.y;
    int threadx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[2][TILE_WIDTH][TILE_WIDTH];

    // prologue
    pipeline.producer_acquire();

    cuda::memcpy_async(
        block,
        &A_s[0][row][col],
        &A[thready * K + (0 * TILE_WIDTH + col)],
        sizeof(float),
        pipeline);

    cuda::memcpy_async(
        block,
        &B_s[0][row][col],
        &B[(0 * TILE_WIDTH + row) * N + threadx],
        sizeof(float),
        pipeline);

    pipeline.producer_commit();

    float sum = 0.0f;
    for(int phase = 0; phase < K / TILE_WIDTH; phase++) {

        int nextPhase = phase + 1;
        int currentStage = phase % 2;
        int nextStage = nextPhase % 2;

        if(nextPhase < K / TILE_WIDTH) {
            pipeline.producer_acquire();

            cuda::memcpy_async(
                block,
                &A_s[nextStage][row][col],
                &A[thready * K +
                   (nextPhase * TILE_WIDTH + col)],
                sizeof(float),
                pipeline);

            cuda::memcpy_async(
                block,
                &B_s[nextStage][row][col],
                &B[(nextPhase * TILE_WIDTH + row) * N +
                   threadx],
                sizeof(float),
                pipeline);

            pipeline.producer_commit();
        }

        __syncthreads();
        pipeline.consumer_wait();

        for(int k = 0; k < TILE_WIDTH; k++) {
            sum += A_s[currentStage][row][k] *
                   B_s[currentStage][k][col];
        }

        pipeline.consumer_release();
        __syncthreads();
    }

    C[thready * N + threadx] = sum;
}

void cuda_gemm_double_buffered(
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
    double_buffer<<<grid, block>>>(A, B, C, M, N, K);
}
