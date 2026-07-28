#pragma once

#include "tensor.h"

struct PicoContext;

// binary operations ======================================

struct PicoTensor* pico_add(struct PicoContext* ctx, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_sub(struct PicoContext* ctx, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_mul(struct PicoContext* ctx, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_matmul(struct PicoContext* ctx, struct PicoTensor* a, struct PicoTensor* b);


// unary operations ======================================

struct PicoTensor* pico_sqrt(struct PicoContext* ctx, struct PicoTensor* a);
struct PicoTensor* pico_sin(struct PicoContext* ctx, struct PicoTensor* a);
struct PicoTensor* pico_cos(struct PicoContext* ctx, struct PicoTensor* a);
struct PicoTensor* pico_tan(struct PicoContext* ctx, struct PicoTensor* a);
struct PicoTensor* pico_tanh(struct PicoContext* ctx, struct PicoTensor* a);
struct PicoTensor* pico_log(struct PicoContext* ctx, struct PicoTensor* a);
