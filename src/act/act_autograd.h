
#pragma once

#include <math.h>

#include "activations.h"
#include "../tensor.h"

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
        parent->grad[i] += self->grad[i] * (self->data[i] * (1 - self->data[i]));
    }
}

static inline void pico_swiglu_backward(struct PicoTensor* self) {
    struct PicoTensor* x = self->parents[0];
    struct PicoTensor* gate = self->parents[1];

    for(int64_t i = 0; i < self->numel; i++) {
        float sig = sigmoid(x->data[i]);
        float silu_grad = sig + x->data[i] * sig * (1.0f - sig);
        x->grad[i] += self->grad[i] * gate->data[i] * silu_grad;
        gate->grad[i] += self->grad[i] * silu(x->data[i]);
    }
}
