#pragma once

// binary operations ======================================

struct PicoTensor* pico_add(struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_sub(struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_mul(struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_matmul(struct PicoTensor* a, struct PicoTensor* b);


// unary operations ======================================

struct PicoTensor* pico_tensor_sqrt(struct PicoTensor* a);
struct PicoTensor* pico_tensor_sin(struct PicoTensor* a);
struct PicoTensor* pico_tensor_cos(struct PicoTensor* a);
struct PicoTensor* pico_tensor_tan(struct PicoTensor* a);
struct PicoTensor* pico_tensor_tanh(struct PicoTensor* a);
struct PicoTensor* pico_tensor_log(struct PicoTensor* a);
