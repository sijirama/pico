#include "../cuda_common.cuh"

__global__ static void pico_cuda_cross_entropy_backward_kernel(const float* __restrict__ logits,
                                                              const float* __restrict__ targets,
                                                              const float* __restrict__ loss_grad,
                                                              float* __restrict__ logits_grad,
                                                              int64_t rows,
                                                              int64_t classes,
                                                              int reduction) {
    int64_t row = blockIdx.x;
    if(row >= rows) {
        return;
    }

    int64_t base = row * classes;
    float max_logit = logits[base];
    for(int64_t c = 1; c < classes; c++) {
        max_logit = fmaxf(max_logit, logits[base + c]);
    }

    float denom = 0.0f;
    for(int64_t c = 0; c < classes; c++) {
        denom += expf(logits[base + c] - max_logit);
    }

    int64_t target = (int64_t)targets[row];
    float scale = (reduction == 1 ? 1.0f : 1.0f / (float)rows) * loss_grad[0];
    for(int64_t c = threadIdx.x; c < classes; c += blockDim.x) {
        float prob = expf(logits[base + c] - max_logit) / denom;
        float one_hot = c == target ? 1.0f : 0.0f;
        logits_grad[base + c] += (prob - one_hot) * scale;
    }
}

extern "C" bool pico_cuda_cross_entropy_backward(struct PicoTensor* self, struct PicoTensor* logits,
                                                 struct PicoTensor* targets, int reduction) {
    if(!pico_cuda_tensor_ready(self, "cross_entropy_backward", "self") ||
       !pico_cuda_tensor_ready(logits, "cross_entropy_backward", "logits") ||
       !pico_cuda_tensor_ready(targets, "cross_entropy_backward", "targets")) {
        return false;
    }
    int64_t classes = logits->shape[logits->ndim - 1];
    int64_t rows = logits->numel / classes;
    pico_cuda_cross_entropy_backward_kernel<<<(unsigned int)rows, 256>>>(logits->data, targets->data, self->grad,
                                                                         logits->grad, rows, classes, reduction);
    return pico_cuda_ok(cudaDeviceSynchronize(), "cross_entropy_backward");
}
