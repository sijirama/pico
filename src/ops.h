#pragma once

#include "arena.h"

// binary operations ======================================

struct PicoTensor* pico_add(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_sub(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_mul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_matmul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);


// unary operations ======================================

struct PicoTensor* pico_tensor_sqrt(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tensor_sin(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tensor_cos(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tensor_tan(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tensor_tanh(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tensor_log(struct Arena* arena, struct PicoTensor* a);
