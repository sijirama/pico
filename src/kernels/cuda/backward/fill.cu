#include "../cuda_common.cuh"

__global__ static void pico_cuda_fill_kernel(float* data, int64_t n, float value) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data[i] = value;
    }
}

static bool pico_cuda_launch_fill(float* data, int64_t n, float value, const char* name) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    pico_cuda_fill_kernel<<<blocks, threads>>>(data, n, value);
    return pico_cuda_ok(cudaDeviceSynchronize(), name);
}

extern "C" bool pico_cuda_fill(struct PicoTensor* tensor, float value) {
    if(!pico_cuda_tensor_ready(tensor, "fill", "tensor")) {
        return false;
    }
    return pico_cuda_launch_fill(tensor->grad, tensor->numel, value, "fill");
}

extern "C" bool pico_cuda_zero_grad(struct PicoTensor* tensor) {
    if(!pico_cuda_tensor_ready(tensor, "zero_grad", "tensor")) {
        return false;
    }
    return pico_cuda_launch_fill(tensor->grad, tensor->numel, 0.0f, "zero_grad");
}
