#include "../cuda_common.cuh"

__global__ static void pico_cuda_grouped_backward_a_kernel(const float* __restrict__ dc,
                                                           const float* __restrict__ b,
                                                           float* __restrict__ da,
                                                           int B,
                                                           int Hq,
                                                           int Hkv,
                                                           int M,
                                                           int N,
                                                           int K,
                                                           int group_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;
    int batch = z / Hq;
    int qh = z % Hq;
    int kvh = qh / group_size;
    if(batch >= B || i >= M || k >= K || kvh >= Hkv) {
        return;
    }

    float acc = 0.0f;
    for(int j = 0; j < N; j++) {
        acc += dc[((batch * Hq + qh) * M + i) * N + j] * b[((batch * Hkv + kvh) * K + k) * N + j];
    }
    da[((batch * Hq + qh) * M + i) * K + k] += acc;
}

__global__ static void pico_cuda_grouped_backward_b_kernel(const float* __restrict__ a,
                                                           const float* __restrict__ dc,
                                                           float* __restrict__ db,
                                                           int B,
                                                           int Hq,
                                                           int Hkv,
                                                           int M,
                                                           int N,
                                                           int K,
                                                           int group_size) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;
    int batch = z / Hkv;
    int kvh = z % Hkv;
    if(batch >= B || k >= K || j >= N) {
        return;
    }

    float acc = 0.0f;
    int q_start = kvh * group_size;
    int q_end = q_start + group_size;
    for(int qh = q_start; qh < q_end && qh < Hq; qh++) {
        for(int i = 0; i < M; i++) {
            acc += a[((batch * Hq + qh) * M + i) * K + k] * dc[((batch * Hq + qh) * M + i) * N + j];
        }
    }
    db[((batch * Hkv + kvh) * K + k) * N + j] += acc;
}

extern "C" bool pico_cuda_grouped_matmul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b,
                                                  int group_size) {
    if(!pico_cuda_tensor_ready(self, "grouped_matmul_backward", "self") ||
       !pico_cuda_tensor_ready(a, "grouped_matmul_backward", "a") ||
       !pico_cuda_tensor_ready(b, "grouped_matmul_backward", "b")) {
        return false;
    }

    int B = (int)a->shape[0];
    int Hq = (int)a->shape[1];
    int Hkv = (int)b->shape[1];
    int M = (int)a->shape[2];
    int K = (int)a->shape[3];
    int N = (int)b->shape[3];
    dim3 block(16, 16);
    pico_cuda_grouped_backward_a_kernel<<<dim3((K + 15) / 16, (M + 15) / 16, B * Hq), block>>>(
        self->grad, b->data, a->grad, B, Hq, Hkv, M, N, K, group_size);
    pico_cuda_grouped_backward_b_kernel<<<dim3((N + 15) / 16, (K + 15) / 16, B * Hkv), block>>>(
        a->data, self->grad, b->grad, B, Hq, Hkv, M, N, K, group_size);
    return pico_cuda_ok(cudaDeviceSynchronize(), "grouped_matmul_backward");
}
