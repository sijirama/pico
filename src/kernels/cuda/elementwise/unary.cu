#include "../cuda_common.cuh"

#include <math.h>

#define PICO_DEFINE_CUDA_UNARY_OP(name, expr)                                      \
__global__ static void pico_cuda_##name##_kernel(const float* input, float* output, int64_t n) { \
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
    if(idx >= n) {                                                                 \
        return;                                                                    \
    }                                                                              \
    float x = input[idx];                                                          \
    output[idx] = (expr);                                                          \
}                                                                                  \
extern "C" bool pico_cuda_##name(struct PicoTensor* a, struct PicoTensor* out) {   \
    if(!pico_cuda_tensor_ready(a, #name, "a") || !pico_cuda_tensor_ready(out, #name, "out")) { \
        return false;                                                              \
    }                                                                              \
    int threads = 256;                                                             \
    int blocks = (int)((out->numel + threads - 1) / threads);                      \
    pico_cuda_##name##_kernel<<<blocks, threads>>>(a->data, out->data, out->numel);\
    return pico_cuda_ok(cudaDeviceSynchronize(), #name);                           \
}

PICO_DEFINE_CUDA_UNARY_OP(sqrt, sqrtf(x))
PICO_DEFINE_CUDA_UNARY_OP(sin, sinf(x))
PICO_DEFINE_CUDA_UNARY_OP(cos, cosf(x))
PICO_DEFINE_CUDA_UNARY_OP(tan, tanf(x))
PICO_DEFINE_CUDA_UNARY_OP(tanh, tanhf(x))
PICO_DEFINE_CUDA_UNARY_OP(log, logf(x))
PICO_DEFINE_CUDA_UNARY_OP(relu, x > 0.0f ? x : 0.0f)
PICO_DEFINE_CUDA_UNARY_OP(sigmoid, 1.0f / (1.0f + expf(-x)))
