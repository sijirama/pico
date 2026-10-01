#include "../cuda_common.cuh"

__global__ static void pico_cuda_swa_matmul_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int batch_count,
                                                   int window) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if(row >= M || col >= N || batch >= batch_count) {
        return;
    }

    float* __restrict__ c = C + batch * M * N;
    if(col > row || col < row - window) {
        c[row * N + col] = 0.0f;
        return;
    }

    const float* __restrict__ a = A + batch * M * K;
    const float* __restrict__ b = B + batch * K * N;

    float acc = 0.0f;
    for(int k = 0; k < K; k++) {
        acc += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = acc;
}

extern "C" bool pico_cuda_swa_matmul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int window) {
    if(!pico_cuda_tensor_ready(a, "swa_matmul", "a") || !pico_cuda_tensor_ready(b, "swa_matmul", "b") ||
       !pico_cuda_tensor_ready(out, "swa_matmul", "out")) {
        return false;
    }

    if(window < 0) {
        fprintf(stderr, "PicoCudaError: swa_matmul window must be non-negative\n");
        return false;
    }

    int M = (int)a->shape[a->ndim - 2];
    int K = (int)a->shape[a->ndim - 1];
    int N = (int)b->shape[b->ndim - 1];

    int batch_count = 1;
    for(int i = 0; i < out->ndim - 2; i++) {
        batch_count *= out->shape[i];
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
    pico_cuda_swa_matmul_kernel<<<grid, block>>>(a->data, b->data, out->data, M, N, K, batch_count, window);
    return pico_cuda_ok(cudaDeviceSynchronize(), "swa_matmul");
}
