#include "../cuda_common.cuh"
#include <math.h>

__global__ static void embedding_f32_kernel(
    const float *__restrict__ token_indices, // Array of input tokens (Size: batch_size * seq_len)
    const float *__restrict__ weight,        // Embedding table (Size: vocab_size * emb_size)
    float *__restrict__ output,              // Target output buffer
    int emb_size,                            // Size of each embedding vector (e.g., 712, 1024)
    int vocab_size) {
    int tx = threadIdx.x; // Dimensional index within the embedding vector
    int bx = blockIdx.x;  // Index of the token being processed

    // Find where the token's weights start in the embedding table
    int token_id = (int)token_indices[bx];
    if(token_id < 0 || token_id >= vocab_size) {
        return;
    }
    int source_offset = token_id * emb_size;

    // Find where this token should write its results in the output array
    int dest_offset = bx * emb_size;

    // Parallel copy: Each thread moves one float (or loops if emb_size > blockDim.x)
    for(int i = tx; i < emb_size; i += blockDim.x) {
        output[dest_offset + i] = weight[source_offset + i];
    }
}

extern "C" bool pico_cuda_embedding(struct PicoTensor *table, struct PicoTensor *input_indices, struct PicoTensor *out) {
    if(!pico_cuda_tensor_ready(table, "embedding", "table") || !pico_cuda_tensor_ready(input_indices, "embedding", "input_indices") ||
       !pico_cuda_tensor_ready(out, "embedding", "out")) {
        return false;
    }

    int threads = 256;
    int blocks = (int)input_indices->numel;

    embedding_f32_kernel<<<blocks, threads>>>(input_indices->data, table->data, out->data, (int)out->shape[1], (int)table->shape[0]);

    return pico_cuda_ok(cudaDeviceSynchronize(), "embedding");
}
