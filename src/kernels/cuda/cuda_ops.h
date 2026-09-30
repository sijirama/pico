#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "../../tensor.h"

struct PicoContext;

#ifdef __cplusplus
extern "C" {
#endif

// INFO: c-visible cuda boundary. cpu code calls these wrappers, and the .cu
// side decides whether tensors need cudaMalloc/cudaMemcpy or can use existing
// device pointers. the c stub keeps normal cpu builds working until cuda is linked.
bool pico_cuda_tensor_alloc(struct PicoContext* ctx, struct PicoTensor* tensor);
bool pico_cuda_tensor_to_cuda(struct PicoContext* ctx, struct PicoTensor* tensor);
bool pico_cuda_tensor_to_cpu(struct PicoContext* ctx, struct PicoTensor* tensor, float* old_data, float* old_grad);

bool pico_cuda_add(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out);
bool pico_cuda_sub(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out);
bool pico_cuda_mul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out);
bool pico_cuda_div(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out);

bool pico_cuda_sqrt(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_sin(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_cos(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_tan(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_tanh(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_log(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_relu(struct PicoTensor* a, struct PicoTensor* out);
bool pico_cuda_sigmoid(struct PicoTensor* a, struct PicoTensor* out);

bool pico_cuda_matmul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out);
bool pico_cuda_grouped_matmul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int group_size);

bool pico_cuda_softmax(struct PicoTensor* input, struct PicoTensor* out, uint8_t dim);
bool pico_cuda_rmsnorm(struct PicoTensor* input, struct PicoTensor* weight, struct PicoTensor* out, float eps);
bool pico_cuda_swiglu(struct PicoTensor* x, struct PicoTensor* gate, struct PicoTensor* out);
bool pico_cuda_embedding(struct PicoTensor* table, struct PicoTensor* input_indices, struct PicoTensor* out);
bool pico_cuda_cross_entropy(struct PicoTensor* logits, struct PicoTensor* targets, struct PicoTensor* out, int reduction);

#ifdef __cplusplus
}
#endif
