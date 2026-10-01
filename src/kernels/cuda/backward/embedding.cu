#include "../cuda_common.cuh"

__global__ static void pico_cuda_embedding_backward_kernel(const float* __restrict__ grad_out,
                                                          const float* __restrict__ input_indices,
                                                          float* __restrict__ table_grad,
                                                          int64_t seq_len,
                                                          int embedding_dim) {
    int64_t token_pos = blockIdx.x;
    int d = threadIdx.x;
    if(token_pos >= seq_len) {
        return;
    }

    int64_t idx = (int64_t)input_indices[token_pos];
    for(int col = d; col < embedding_dim; col += blockDim.x) {
        atomicAdd(&table_grad[idx * embedding_dim + col], grad_out[token_pos * embedding_dim + col]);
    }
}

extern "C" bool pico_cuda_embedding_backward(struct PicoTensor* self, struct PicoTensor* table,
                                             struct PicoTensor* input_indices) {
    if(!pico_cuda_tensor_ready(self, "embedding_backward", "self") ||
       !pico_cuda_tensor_ready(table, "embedding_backward", "table") ||
       !pico_cuda_tensor_ready(input_indices, "embedding_backward", "input_indices")) {
        return false;
    }

    int64_t seq_len = input_indices->shape[0];
    int embedding_dim = (int)table->shape[1];
    pico_cuda_embedding_backward_kernel<<<(unsigned int)seq_len, 256>>>(self->grad, input_indices->data, table->grad,
                                                                       seq_len, embedding_dim);
    return pico_cuda_ok(cudaDeviceSynchronize(), "embedding_backward");
}
