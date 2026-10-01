#include "../cuda_common.cuh"

__device__ static inline float pico_cuda_random_unit(uint32_t seed, int64_t idx) {
    uint32_t x = seed ^ (uint32_t)idx;
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;

    return (float)(x & 0x00ffffffU) / 16777216.0f;
}

__global__ static void pico_cuda_fused_swiglu_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ mask,
    float *__restrict__ out,
    int64_t n,
    float dropout_p,
    bool training,
    uint32_t seed) {

    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) {
        return;
    }

    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    float keep_scale = 1.0f;

    if(training && dropout_p > 0.0f) {
        float r = pico_cuda_random_unit(seed, i);
        keep_scale = r >= dropout_p ? 1.0f / (1.0f - dropout_p) : 0.0f;
    }

    mask[i] = keep_scale;
    out[i] = silu * up[i] * keep_scale;
}

extern "C" bool pico_cuda_fused_swiglu(
    struct PicoTensor *gate,
    struct PicoTensor *up,
    struct PicoTensor *mask,
    struct PicoTensor *out,
    float dropout_p,
    bool training,
    uint32_t seed) {
    if(!pico_cuda_tensor_ready(gate, "fused_swiglu", "gate") || !pico_cuda_tensor_ready(up, "fused_swiglu", "up") ||
       !pico_cuda_tensor_ready(mask, "fused_swiglu", "mask") || !pico_cuda_tensor_ready(out, "fused_swiglu", "out")) {
        return false;
    }

    if(gate->numel != up->numel || gate->numel != mask->numel || gate->numel != out->numel) {
        fprintf(stderr, "PicoCudaError: fused_swiglu inputs, mask, and output must have matching numel\n");
        return false;
    }

    if(dropout_p < 0.0f || dropout_p >= 1.0f) {
        fprintf(stderr, "PicoCudaError: fused_swiglu dropout_p must be in [0, 1)\n");
        return false;
    }

    int threads = 256;
    int blocks = (int)((out->numel + threads - 1) / threads);
    pico_cuda_fused_swiglu_kernel<<<blocks, threads>>>(gate->data, up->data, mask->data, out->data, out->numel, dropout_p, training, seed);
    return pico_cuda_ok(cudaDeviceSynchronize(), "fused_swiglu");
}
