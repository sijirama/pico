#include "../cuda_common.cuh"

__device__ static inline int64_t pico_cuda_parent_index(int64_t i, int64_t parent_numel, int64_t out_numel) {
    if(parent_numel == out_numel) {
        return i;
    }
    if(parent_numel == 1) {
        return 0;
    }
    return i % parent_numel;
}

__global__ static void pico_cuda_binary_backward_kernel(const float* __restrict__ grad,
                                                        const float* __restrict__ a_data,
                                                        const float* __restrict__ b_data,
                                                        float* __restrict__ a_grad,
                                                        float* __restrict__ b_grad,
                                                        int64_t out_numel,
                                                        int64_t a_numel,
                                                        int64_t b_numel,
                                                        int op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= out_numel) {
        return;
    }

    int64_t ia = pico_cuda_parent_index(i, a_numel, out_numel);
    int64_t ib = pico_cuda_parent_index(i, b_numel, out_numel);
    float g = grad[i];
    float da = 0.0f;
    float db = 0.0f;

    if(op == 0) {
        da = g;
        db = g;
    } else if(op == 1) {
        da = g;
        db = -g;
    } else if(op == 2) {
        da = g * b_data[ib];
        db = g * a_data[ia];
    } else {
        float denom = b_data[ib];
        da = g / denom;
        db = -g * a_data[ia] / (denom * denom);
    }

    atomicAdd(&a_grad[ia], da);
    atomicAdd(&b_grad[ib], db);
}

static bool pico_cuda_binary_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b, int op,
                                      const char* name) {
    if(!pico_cuda_tensor_ready(self, name, "self") || !pico_cuda_tensor_ready(a, name, "a") ||
       !pico_cuda_tensor_ready(b, name, "b")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((self->numel + threads - 1) / threads);
    pico_cuda_binary_backward_kernel<<<blocks, threads>>>(self->grad, a->data, b->data, a->grad, b->grad,
                                                         self->numel, a->numel, b->numel, op);
    return pico_cuda_ok(cudaDeviceSynchronize(), name);
}

extern "C" bool pico_cuda_add_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b) {
    return pico_cuda_binary_backward(self, a, b, 0, "add_backward");
}

extern "C" bool pico_cuda_sub_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b) {
    return pico_cuda_binary_backward(self, a, b, 1, "sub_backward");
}

extern "C" bool pico_cuda_mul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b) {
    return pico_cuda_binary_backward(self, a, b, 2, "mul_backward");
}

extern "C" bool pico_cuda_div_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b) {
    return pico_cuda_binary_backward(self, a, b, 3, "div_backward");
}
