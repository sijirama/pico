#include "../cuda_common.cuh"

__global__ static void pico_cuda_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K,
                                               int batch_count, int b_batch_count) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if(row >= M || col >= N || batch >= batch_count) {
        return;
    }

    int b_batch = b_batch_count == 1 ? 0 : batch;
    const float* a = A + batch * M * K;
    const float* b = B + b_batch * K * N;
    float* c = C + batch * M * N;

    float acc = 0.0f;
    for(int k = 0; k < K; k++) {
        acc += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = acc;
}

extern "C" bool pico_cuda_matmul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    if(!pico_cuda_tensor_ready(a, "matmul", "a") || !pico_cuda_tensor_ready(b, "matmul", "b") ||
       !pico_cuda_tensor_ready(out, "matmul", "out")) {
        return false;
    }

    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    int batch_count = 1;
    for(int i = 0; i < out->ndim - 2; i++) {
        batch_count *= out->shape[i];
    }

    int b_batch_count = 1;
    if(b->ndim > 2) {
        for(int i = 0; i < b->ndim - 2; i++) {
            b_batch_count *= b->shape[i];
        }
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count);
    pico_cuda_matmul_kernel<<<grid, block>>>(a->data, b->data, out->data, M, N, K, batch_count, b_batch_count);
    return pico_cuda_ok(cudaDeviceSynchronize(), "matmul");
}
