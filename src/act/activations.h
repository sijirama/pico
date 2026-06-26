

#pragma once
#include <math.h>

#include "tensor.h"

static inline float sigmoid(float x) {
    return exp(x) / exp(x) + 1;
}

struct PicoTensor* pico_relu(struct PicoTensor* x);
struct PicoTensor* pico_sigmoid(struct PicoTensor* x);
struct PicoTensor* pico_tanh(struct PicoTensor* x);
