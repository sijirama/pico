#pragma once

#include <stdint.h>

#include "ctx.h"
#include "tensor.h"

// INFO: tensor ops are structural helpers, not math/autograd ops. this is where
// shape/view/copy utilities should live: clone, transpose, reshape, cat, etc.

void pico_transpose_2d(struct PicoTensor* tensor);
struct PicoTensor* pico_cat(struct PicoContext* ctx, struct PicoTensor* a, struct PicoTensor* b, int dim);

struct PicoTensor* pico_clone(struct PicoContext* ctx, struct PicoTensor* tensor);
void pico_view(struct PicoContext* ctx, struct PicoTensor* tensor, int64_t* shape, int ndim);
void pico_permute(struct PicoContext* ctx, struct PicoTensor* tensor, int64_t* axes);
struct PicoTensor* pico_dropout(struct PicoContext* ctx, struct PicoTensor* tensor, float p);
struct PicoTensor* pico_softmax(struct PicoContext* ctx, struct PicoTensor* tensor, uint8_t dim);
struct PicoTensor* pico_causal_softmax(struct PicoContext* ctx, struct PicoTensor* tensor, uint8_t dim, int window);
struct PicoTensor* pico_sum(struct PicoContext* ctx, struct PicoTensor* tensor, int dim);
struct PicoTensor* pico_mean(struct PicoContext* ctx, struct PicoTensor* tensor, int dim);
struct PicoTensor* pico_var(struct PicoContext* ctx, struct PicoTensor* tensor, int dim);
void pico_reshape(struct PicoTensor* tensor, int64_t* shape, int ndim);
