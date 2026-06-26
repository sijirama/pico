
#pragma once

#include <math.h>

#include "activations.h"
#include "tensor.h"

static inline void pico_relu_backward(struct PicoTensor* self) {
    struct PicoTensor* parent = self->parents[0];

    int64_t N = parent->numel;

    for(int64_t i = 0; i < N; i++) {
        parent->grad[i] += self->grad[i] * (self->data[i] > 0);
    }
}

static inline void pico_sigmoid_backward(struct PicoTensor* self) {
    struct PicoTensor* parent = self->parents[0];

    int64_t N = parent->numel;

    for(int64_t i = 0; i < N; i++) {
        parent->grad[i] += self->grad[i] * (sigmoid(self->data[i]) * (1 - sigmoid(self->data[i])));
    }
}

static inline void pico_tanh_backward(struct PicoTensor* self) {
    struct PicoTensor* parent = self->parents[0];

    int64_t N = parent->numel;

    for(int64_t i = 0; i < N; i++) {
        parent->grad[i] += self->grad[i] * (1 - powf(tanh(self->data[i]), 2.0f));
    }
}
