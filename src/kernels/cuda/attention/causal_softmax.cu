#include "../cuda_common.cuh"

#include <math.h>

__global__ static void pico_cuda_causal_softmax_lastdim_kernel(const float* __restrict__ input,
                                                               float* __restrict__ output,
                                                               int64_t rows,
                                                               int64_t q_len,
                                                               int64_t k_len,
                                                               int window) {
    int64_t row = blockIdx.x;
    if(row >= rows) {
        return;
    }

    int64_t q = row % q_len;
    int64_t base = row * k_len;
    int64_t start_k = window < 0 ? 0 : q - window;
    if(start_k < 0) {
        start_k = 0;
    }
    int64_t end_k = q < k_len - 1 ? q : k_len - 1;

    float max_value = -INFINITY;
    for(int64_t k = start_k; k <= end_k; k++) {
        float value = input[base + k];
        if(value > max_value) {
            max_value = value;
        }
    }

    float sum = 0.0f;
    for(int64_t k = threadIdx.x; k < k_len; k += blockDim.x) {
        bool allowed = k >= start_k && k <= end_k;
        float value = allowed ? expf(input[base + k] - max_value) : 0.0f;
        output[base + k] = value;
    }
    __syncthreads();

    for(int64_t k = 0; k < k_len; k++) {
        sum += output[base + k];
    }

    if(sum == 0.0f) {
        return;
    }

    for(int64_t k = threadIdx.x; k < k_len; k += blockDim.x) {
        output[base + k] /= sum;
    }
}

extern "C" bool pico_cuda_causal_softmax(struct PicoTensor* input, struct PicoTensor* out, uint8_t dim, int window) {
    if(!pico_cuda_tensor_ready(input, "causal_softmax", "input") ||
       !pico_cuda_tensor_ready(out, "causal_softmax", "out")) {
        return false;
    }

    if(input->ndim != 3 && input->ndim != 4) {
        fprintf(stderr, "PicoCudaError: causal_softmax expects 3D or 4D attention scores\n");
        return false;
    }

    if(dim != input->ndim - 1) {
        fprintf(stderr, "PicoCudaError: causal_softmax currently supports last dim only\n");
        return false;
    }

    int64_t q_len = input->shape[input->ndim - 2];
    int64_t k_len = input->shape[input->ndim - 1];
    if(q_len != k_len) {
        fprintf(stderr, "PicoCudaError: causal_softmax expects square attention scores\n");
        return false;
    }

    int64_t rows = input->numel / k_len;
    pico_cuda_causal_softmax_lastdim_kernel<<<(unsigned int)rows, 256>>>(input->data, out->data, rows, q_len, k_len,
                                                                        window);
    return pico_cuda_ok(cudaDeviceSynchronize(), "causal_softmax");
}
