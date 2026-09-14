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
struct PicoTensor* pico_softmax(struct PicoContext* ctx, struct PicoTensor* tensor, uint8_t dim);
void pico_reshape(struct PicoTensor* tensor, int64_t* shape, int ndim);
