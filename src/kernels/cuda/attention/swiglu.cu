#include "../cuda_common.cuh"

#include <math.h>

__device__ static float pico_cuda_silu(float input) {
    return input / (1.0f + expf(-input));
}

__global__ static void pico_cuda_swiglu_kernel(const float* __restrict__ x, const float* __restrict__ gate,
                                               float* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) {
        return;
    }
    out[idx] = pico_cuda_silu(x[idx]) * gate[idx];
}

extern "C" bool pico_cuda_swiglu(struct PicoTensor* x, struct PicoTensor* gate, struct PicoTensor* out) {
    if(!pico_cuda_tensor_ready(x, "swiglu", "x") || !pico_cuda_tensor_ready(gate, "swiglu", "gate") ||
       !pico_cuda_tensor_ready(out, "swiglu", "out")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)((out->numel + threads - 1) / threads);
    pico_cuda_swiglu_kernel<<<blocks, threads>>>(x->data, gate->data, out->data, out->numel);
    return pico_cuda_ok(cudaDeviceSynchronize(), "swiglu");
}
