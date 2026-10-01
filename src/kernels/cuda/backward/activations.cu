#include "../cuda_common.cuh"

__global__ static void pico_cuda_relu_backward_kernel(
    const float *__restrict__ self_grad, const float *__restrict__ parent_data, float *__restrict__ parent_grad, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        parent_grad[i] += parent_data[i] > 0.0f ? self_grad[i] : 0.0f;
    }
}

extern "C" bool pico_cuda_relu_backward(struct PicoTensor *self, struct PicoTensor *parent) {
    if(!pico_cuda_tensor_ready(self, "relu_backward", "self") || !pico_cuda_tensor_ready(parent, "relu_backward", "parent")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((self->numel + threads - 1) / threads);
    pico_cuda_relu_backward_kernel<<<blocks, threads>>>(self->grad, parent->data, parent->grad, self->numel);
    return pico_cuda_ok(cudaDeviceSynchronize(), "relu_backward");
}

__device__ static inline float pico_cuda_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ static void pico_cuda_swiglu_backward_kernel(
    const float *__restrict__ self_grad,
    const float *__restrict__ gate_data,
    const float *__restrict__ up_data,
    const float *__restrict__ mask_data,
    float *__restrict__ gate_grad,
    float *__restrict__ up_grad,
    int64_t n,
    bool has_mask) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) {
        return;
    }

    float sig = pico_cuda_sigmoidf(gate_data[i]);
    float silu_gate = gate_data[i] * sig;
    float silu_grad = sig + gate_data[i] * sig * (1.0f - sig);
    float upstream = self_grad[i] * (has_mask ? mask_data[i] : 1.0f);

    gate_grad[i] += upstream * up_data[i] * silu_grad;
    up_grad[i] += upstream * silu_gate;
}

extern "C" bool pico_cuda_swiglu_backward(struct PicoTensor *self, struct PicoTensor *gate, struct PicoTensor *up) {
    if(!pico_cuda_tensor_ready(self, "swiglu_backward", "self") || !pico_cuda_tensor_ready(gate, "swiglu_backward", "gate") ||
       !pico_cuda_tensor_ready(up, "swiglu_backward", "up")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((self->numel + threads - 1) / threads);
    pico_cuda_swiglu_backward_kernel<<<blocks, threads>>>(
        self->grad, gate->data, up->data, nullptr, gate->grad, up->grad, self->numel, false);
    return pico_cuda_ok(cudaDeviceSynchronize(), "swiglu_backward");
}

extern "C" bool
pico_cuda_fused_swiglu_backward(struct PicoTensor *self, struct PicoTensor *gate, struct PicoTensor *up, struct PicoTensor *mask) {
    if(!pico_cuda_tensor_ready(self, "fused_swiglu_backward", "self") || !pico_cuda_tensor_ready(gate, "fused_swiglu_backward", "gate") ||
       !pico_cuda_tensor_ready(up, "fused_swiglu_backward", "up") || !pico_cuda_tensor_ready(mask, "fused_swiglu_backward", "mask")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((self->numel + threads - 1) / threads);
    pico_cuda_swiglu_backward_kernel<<<blocks, threads>>>(
        self->grad, gate->data, up->data, mask->data, gate->grad, up->grad, self->numel, true);
    return pico_cuda_ok(cudaDeviceSynchronize(), "fused_swiglu_backward");
}
