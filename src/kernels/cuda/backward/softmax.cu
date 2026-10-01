#include "../cuda_common.cuh"

__global__ static void pico_cuda_softmax_backward_lastdim_kernel(const float* __restrict__ self_data,
                                                                 const float* __restrict__ self_grad,
                                                                 float* __restrict__ input_grad,
                                                                 int64_t rows,
                                                                 int64_t axis_len) {
    int64_t row = blockIdx.x;
    if(row >= rows) {
        return;
    }

    float dot = 0.0f;
    int64_t base = row * axis_len;
    for(int64_t i = 0; i < axis_len; i++) {
        dot += self_grad[base + i] * self_data[base + i];
    }

    for(int64_t i = threadIdx.x; i < axis_len; i += blockDim.x) {
        float s = self_data[base + i];
        input_grad[base + i] += s * (self_grad[base + i] - dot);
    }
}

extern "C" bool pico_cuda_softmax_backward(struct PicoTensor* self, struct PicoTensor* input, int dim) {
    if(!pico_cuda_tensor_ready(self, "softmax_backward", "self") ||
       !pico_cuda_tensor_ready(input, "softmax_backward", "input")) {
        return false;
    }

    if(dim != self->ndim - 1) {
        fprintf(stderr, "PicoCudaError: softmax_backward currently supports last dim only\n");
        return false;
    }

    int64_t axis_len = self->shape[dim];
    int64_t rows = self->numel / axis_len;
    pico_cuda_softmax_backward_lastdim_kernel<<<(unsigned int)rows, 256>>>(self->data, self->grad, input->grad, rows,
                                                                          axis_len);
    return pico_cuda_ok(cudaDeviceSynchronize(), "softmax_backward");
}
