#include "../cuda_common.cuh"

__global__ static void pico_cuda_swa_matmul_backward_a_kernel(const float* __restrict__ dc,
                                                              const float* __restrict__ b,
                                                              float* __restrict__ da,
                                                              int batch_count,
                                                              int M,
                                                              int N,
                                                              int K,
                                                              int window) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    if(batch >= batch_count || i >= M || k >= K) {
        return;
    }

    float acc = 0.0f;
    int start_j = i - window;
    if(start_j < 0) {
        start_j = 0;
    }
    int end_j = i < N - 1 ? i : N - 1;

    for(int j = start_j; j <= end_j; j++) {
        acc += dc[batch * M * N + i * N + j] * b[batch * K * N + k * N + j];
    }
    da[batch * M * K + i * K + k] += acc;
}

__global__ static void pico_cuda_swa_matmul_backward_b_kernel(const float* __restrict__ a,
                                                              const float* __restrict__ dc,
                                                              float* __restrict__ db,
                                                              int batch_count,
                                                              int M,
                                                              int N,
                                                              int K,
                                                              int window) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    if(batch >= batch_count || k >= K || j >= N) {
        return;
    }

    float acc = 0.0f;
    int start_i = j;
    int end_i = j + window;
    if(end_i > M - 1) {
        end_i = M - 1;
    }

    for(int i = start_i; i <= end_i; i++) {
        acc += a[batch * M * K + i * K + k] * dc[batch * M * N + i * N + j];
    }
    db[batch * K * N + k * N + j] += acc;
}

extern "C" bool pico_cuda_swa_matmul_backward(struct PicoTensor* self,
                                              struct PicoTensor* a,
                                              struct PicoTensor* b,
                                              int window) {
    if(!pico_cuda_tensor_ready(self, "swa_matmul_backward", "self") ||
       !pico_cuda_tensor_ready(a, "swa_matmul_backward", "a") ||
       !pico_cuda_tensor_ready(b, "swa_matmul_backward", "b")) {
        return false;
    }

    if(window < 0) {
        fprintf(stderr, "PicoCudaError: swa_matmul_backward window must be non-negative\n");
        return false;
    }

    int M = (int)a->shape[a->ndim - 2];
    int K = (int)a->shape[a->ndim - 1];
    int N = (int)b->shape[b->ndim - 1];
    int batch_count = (int)(self->numel / (M * N));

    dim3 block(16, 16);
    dim3 grid_a((K + 15) / 16, (M + 15) / 16, batch_count);
    dim3 grid_b((N + 15) / 16, (K + 15) / 16, batch_count);
    pico_cuda_swa_matmul_backward_a_kernel<<<grid_a, block>>>(self->grad, b->data, a->grad, batch_count, M, N, K,
                                                              window);
    pico_cuda_swa_matmul_backward_b_kernel<<<grid_b, block>>>(a->data, self->grad, b->grad, batch_count, M, N, K,
                                                              window);
    return pico_cuda_ok(cudaDeviceSynchronize(), "swa_matmul_backward");
}
