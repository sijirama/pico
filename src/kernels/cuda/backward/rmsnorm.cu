#include "../cuda_common.cuh"

__global__ static void pico_cuda_rmsnorm_backward_kernel(const float* __restrict__ grad,
                                                         const float* __restrict__ input,
                                                         const float* __restrict__ weight,
                                                         const float* __restrict__ eps,
                                                         float* __restrict__ input_grad,
                                                         float* __restrict__ weight_grad,
                                                         int64_t rows,
                                                         int D) {
    int64_t row = blockIdx.x;
    if(row >= rows) {
        return;
    }

    int64_t base = row * D;
    float mean_sq = 0.0f;
    for(int d = 0; d < D; d++) {
        float x = input[base + d];
        mean_sq += x * x;
    }
    mean_sq /= (float)D;
    float inv_rms = rsqrtf(mean_sq + eps[0]);
    float inv_rms_cubed = inv_rms * inv_rms * inv_rms;

    float dot = 0.0f;
    for(int d = 0; d < D; d++) {
        dot += grad[base + d] * weight[d] * input[base + d];
    }

    for(int d = threadIdx.x; d < D; d += blockDim.x) {
        float upstream_times_weight = grad[base + d] * weight[d];
        input_grad[base + d] += upstream_times_weight * inv_rms - input[base + d] * dot * inv_rms_cubed / (float)D;
        atomicAdd(&weight_grad[d], grad[base + d] * input[base + d] * inv_rms);
    }
}

extern "C" bool pico_cuda_rmsnorm_backward(struct PicoTensor* self, struct PicoTensor* input, struct PicoTensor* weight,
                                           struct PicoTensor* eps) {
    if(!pico_cuda_tensor_ready(self, "rmsnorm_backward", "self") ||
       !pico_cuda_tensor_ready(input, "rmsnorm_backward", "input") ||
       !pico_cuda_tensor_ready(weight, "rmsnorm_backward", "weight") ||
       !pico_cuda_tensor_ready(eps, "rmsnorm_backward", "eps")) {
        return false;
    }

    int D = (int)input->shape[input->ndim - 1];
    int64_t rows = input->numel / D;
    pico_cuda_rmsnorm_backward_kernel<<<(unsigned int)rows, 256>>>(self->grad, input->data, weight->data, eps->data,
                                                                  input->grad, weight->grad, rows, D);
    return pico_cuda_ok(cudaDeviceSynchronize(), "rmsnorm_backward");
}
