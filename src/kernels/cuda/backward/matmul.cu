#include "../cuda_common.cuh"

__global__ static void pico_cuda_matmul_backward_a_kernel(const float* __restrict__ dc,
                                                          const float* __restrict__ b,
                                                          float* __restrict__ da,
                                                          int batch_count,
                                                          int M,
                                                          int N,
                                                          int K,
                                                          int b_batch_count) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    if(batch >= batch_count || i >= M || k >= K) {
        return;
    }

    int b_batch = b_batch_count == 1 ? 0 : batch;
    float acc = 0.0f;
    for(int j = 0; j < N; j++) {
        acc += dc[batch * M * N + i * N + j] * b[b_batch * K * N + k * N + j];
    }
    da[batch * M * K + i * K + k] += acc;
}

__global__ static void pico_cuda_matmul_backward_b_kernel(const float* __restrict__ a,
                                                          const float* __restrict__ dc,
                                                          float* __restrict__ db,
                                                          int batch_count,
                                                          int M,
                                                          int N,
                                                          int K,
                                                          int b_batch_count) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b_batch = blockIdx.z;
    if(b_batch >= b_batch_count || k >= K || j >= N) {
        return;
    }

    float acc = 0.0f;
    for(int batch = 0; batch < batch_count; batch++) {
        if(b_batch_count != 1 && batch != b_batch) {
            continue;
        }
        for(int i = 0; i < M; i++) {
            acc += a[batch * M * K + i * K + k] * dc[batch * M * N + i * N + j];
        }
    }
    db[b_batch * K * N + k * N + j] += acc;
}

extern "C" bool pico_cuda_matmul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_cuda_tensor_ready(self, "matmul_backward", "self") ||
       !pico_cuda_tensor_ready(a, "matmul_backward", "a") || !pico_cuda_tensor_ready(b, "matmul_backward", "b")) {
        return false;
    }

    int M = (int)a->shape[a->ndim - 2];
    int K = (int)a->shape[a->ndim - 1];
    int N = (int)b->shape[b->ndim - 1];
    int batch_count = (int)(self->numel / (M * N));
    int b_batch_count = b->ndim > 2 ? (int)(b->numel / (K * N)) : 1;

    dim3 block(16, 16);
    dim3 grid_a((K + 15) / 16, (M + 15) / 16, batch_count);
    dim3 grid_b((N + 15) / 16, (K + 15) / 16, b_batch_count);
    pico_cuda_matmul_backward_a_kernel<<<grid_a, block>>>(self->grad, b->data, a->grad, batch_count, M, N, K,
                                                          b_batch_count);
    pico_cuda_matmul_backward_b_kernel<<<grid_b, block>>>(a->data, self->grad, b->grad, batch_count, M, N, K,
                                                          b_batch_count);
    return pico_cuda_ok(cudaDeviceSynchronize(), "matmul_backward");
}
