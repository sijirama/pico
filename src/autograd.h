/*
 *

 _backward is called on a specific tensor —
 say it's called on b (where b = x * y). Its job is:
 push gradient from b backward into x and y (writing into x->grad and y->grad).


 To do that, it needs to know: "how much gradient has b itself accumulated so far?" —
 that's the "upstream gradient"

 So: self->grad is the input to the formula (what b has accumulated).
 x->grad and y->grad are what get written to (the outputs, what gets pushed to parents).

 * */

#pragma once

#include "tensor.h"

static inline void pico_add_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];
    struct PicoTensor* b = self->parents[1];

    for (int64_t i = 0; i < self->numel; i++) {
        a->grad[i] += self->grad[i];
        b->grad[i] += self->grad[i];
    }
}

static inline void pico_sub_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];
    struct PicoTensor* b = self->parents[1];

    for (int64_t i = 0; i < self->numel; i++) {
        a->grad[i] += self->grad[i];
        b->grad[i] -= self->grad[i];
    }
}
