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
bool pico_cuda_rsqrt(struct PicoTensor* a, struct PicoTensor* out);
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
bool pico_cuda_mean(struct PicoTensor* input, struct PicoTensor* out, int dim);
bool pico_cuda_rmsnorm(struct PicoTensor* input, struct PicoTensor* weight, struct PicoTensor* out, float eps);
bool pico_cuda_swiglu(struct PicoTensor* x, struct PicoTensor* gate, struct PicoTensor* out);
bool pico_cuda_fused_swiglu(struct PicoTensor* gate, struct PicoTensor* up, struct PicoTensor* mask,
                            struct PicoTensor* out, float dropout_p, bool training, uint32_t seed);
bool pico_cuda_embedding(struct PicoTensor* table, struct PicoTensor* input_indices, struct PicoTensor* out);
bool pico_cuda_cross_entropy(struct PicoTensor* logits, struct PicoTensor* targets, struct PicoTensor* out, int reduction);

// ==================== backward / optimizer helpers
bool pico_cuda_fill(struct PicoTensor* tensor, float value);
bool pico_cuda_zero_grad(struct PicoTensor* tensor);

bool pico_cuda_add_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b);
bool pico_cuda_sub_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b);
bool pico_cuda_mul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b);
bool pico_cuda_div_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b);
bool pico_cuda_relu_backward(struct PicoTensor* self, struct PicoTensor* parent);
bool pico_cuda_swiglu_backward(struct PicoTensor* self, struct PicoTensor* gate, struct PicoTensor* up);
bool pico_cuda_fused_swiglu_backward(struct PicoTensor* self, struct PicoTensor* gate, struct PicoTensor* up,
                                     struct PicoTensor* mask);
bool pico_cuda_softmax_backward(struct PicoTensor* self, struct PicoTensor* input, int dim);
bool pico_cuda_matmul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b);
bool pico_cuda_grouped_matmul_backward(struct PicoTensor* self, struct PicoTensor* a, struct PicoTensor* b,
                                       int group_size);
bool pico_cuda_cross_entropy_backward(struct PicoTensor* self, struct PicoTensor* logits, struct PicoTensor* targets,
                                      int reduction);
bool pico_cuda_rmsnorm_backward(struct PicoTensor* self, struct PicoTensor* input, struct PicoTensor* weight,
                                struct PicoTensor* eps);
bool pico_cuda_embedding_backward(struct PicoTensor* self, struct PicoTensor* table, struct PicoTensor* input_indices);

bool pico_cuda_optim_alloc(float** ptr, int64_t numel);
bool pico_cuda_optim_free(float* ptr);
bool pico_cuda_adamw_step(struct PicoTensor* tensor, float* m, float* v, float lr, float beta1, float beta2,
                          float eps, float weight_decay, float beta1_correction, float beta2_correction);

#ifdef __cplusplus
}
#endif
