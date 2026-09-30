#include "cuda_ops.h"

#include <stdio.h>

extern "C" bool pico_cuda_grouped_matmul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int group_size) {
    (void)a; (void)b; (void)out; (void)group_size;
    fprintf(stderr, "PicoCudaError: grouped_matmul CUDA kernel is not implemented yet\n");
    return false;
}

extern "C" bool pico_cuda_embedding(struct PicoTensor* table, struct PicoTensor* input_indices, struct PicoTensor* out) {
    (void)table; (void)input_indices; (void)out;
    fprintf(stderr, "PicoCudaError: embedding CUDA kernel is not implemented yet\n");
    return false;
}

extern "C" bool pico_cuda_cross_entropy(struct PicoTensor* logits, struct PicoTensor* targets, struct PicoTensor* out, int reduction) {
    (void)logits; (void)targets; (void)out; (void)reduction;
    fprintf(stderr, "PicoCudaError: cross_entropy CUDA kernel is not implemented yet\n");
    return false;
}
