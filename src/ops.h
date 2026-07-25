#pragma once

#include "arena.h"

// binary operations ======================================

struct PicoTensor* pico_add(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_sub(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_mul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_matmul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b);


// unary operations ======================================

struct PicoTensor* pico_sqrt(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_sin(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_cos(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tan(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_tanh(struct Arena* arena, struct PicoTensor* a);
struct PicoTensor* pico_log(struct Arena* arena, struct PicoTensor* a);
