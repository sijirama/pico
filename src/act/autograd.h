
#pragma once

#include "tensor.h"

static inline void pico_relu_backward(struct PicoTensor* self) {
    struct PicoTensor* parent = self->parents[0];

    int64_t N = parent->numel;

    for(int64_t i = 0; i < N; i++) {
        parent->grad[i] += self->grad[i] * (self->data[i] > 0);
    }
}
