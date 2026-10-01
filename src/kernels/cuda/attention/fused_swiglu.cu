#include "../cuda_common.cuh"

__global__ static void pico_cuda_fused_swiglu_kernel(
    const float *__restrict__ x, const float *__restrict__ gate, float *__restrict__ out, int64_t n, float dropout_p, bool training) {

    (void)x;
    (void)gate;
    (void)out;
    (void)n;
    (void)dropout_p;
    (void)training;
}

extern "C" bool
pico_cuda_fused_swiglu(struct PicoTensor *x, struct PicoTensor *gate, struct PicoTensor *out, float dropout_p, bool training) {
    if(!pico_cuda_tensor_ready(x, "fused_swiglu", "x") || !pico_cuda_tensor_ready(gate, "fused_swiglu", "gate") ||
       !pico_cuda_tensor_ready(out, "fused_swiglu", "out")) {
        return false;
    }

    if(x->numel != gate->numel || x->numel != out->numel) {
        fprintf(stderr, "PicoCudaError: fused_swiglu inputs and output must have matching numel\n");
        return false;
    }

    if(dropout_p < 0.0f || dropout_p >= 1.0f) {
        fprintf(stderr, "PicoCudaError: fused_swiglu dropout_p must be in [0, 1)\n");
        return false;
    }

    (void)training;
    fprintf(stderr, "PicoCudaError: fused_swiglu CUDA kernel is not implemented yet\n");
    return false;

    int threads = 256;
    int blocks = (int)((out->numel + threads - 1) / threads);
    pico_cuda_fused_swiglu_kernel<<<blocks, threads>>>(x->data, gate->data, out->data, out->numel, dropout_p, training);
    return pico_cuda_ok(cudaDeviceSynchronize(), "fused_swiglu");
}
