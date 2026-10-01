#include "../cuda_common.cuh"

extern "C" bool pico_cuda_optim_alloc(float** ptr, int64_t numel) {
    if(ptr == nullptr || numel < 0) {
        return false;
    }

    if(!pico_cuda_ok(cudaMalloc((void**)ptr, (size_t)numel * sizeof(float)), "optim state allocation")) {
        return false;
    }

    return pico_cuda_ok(cudaMemset(*ptr, 0, (size_t)numel * sizeof(float)), "optim state reset");
}

extern "C" bool pico_cuda_optim_free(float* ptr) {
    if(ptr == nullptr) {
        return true;
    }
    return pico_cuda_ok(cudaFree(ptr), "optim state free");
}

__global__ static void pico_cuda_adamw_step_kernel(float* __restrict__ data,
                                                   const float* __restrict__ grad,
                                                   float* __restrict__ m,
                                                   float* __restrict__ v,
                                                   int64_t n,
                                                   float lr,
                                                   float beta1,
                                                   float beta2,
                                                   float eps,
                                                   float weight_decay,
                                                   float beta1_correction,
                                                   float beta2_correction) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) {
        return;
    }

    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat = m[i] / beta1_correction;
    float v_hat = v[i] / beta2_correction;
    data[i] -= lr * weight_decay * data[i];
    data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

extern "C" bool pico_cuda_adamw_step(struct PicoTensor* tensor,
                                     float* m,
                                     float* v,
                                     float lr,
                                     float beta1,
                                     float beta2,
                                     float eps,
                                     float weight_decay,
                                     float beta1_correction,
                                     float beta2_correction) {
    if(!pico_cuda_tensor_ready(tensor, "adamw_step", "tensor") || m == nullptr || v == nullptr) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((tensor->numel + threads - 1) / threads);
    pico_cuda_adamw_step_kernel<<<blocks, threads>>>(tensor->data, tensor->grad, m, v, tensor->numel, lr, beta1,
                                                    beta2, eps, weight_decay, beta1_correction, beta2_correction);
    return pico_cuda_ok(cudaDeviceSynchronize(), "adamw_step");
}

