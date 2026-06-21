#pragma once

#include "tensor.h"

// scalar (no SIMD) element-wise add with broadcasting.
// out is pre-allocated by the op with the broadcasted shape; we just fill it.
// map_index handles the broadcast: for each output element, it finds which
// element of a and of b to read (reusing along stretched/size-1 dims).
static inline void pico_add_cpu_scalar(struct PicoTensor* a, struct PicoTensor* b,
                                       struct PicoTensor* out) {
    for(int64_t i = 0; i < out->numel; i++) {
        int64_t ia = map_index(i, a, out->strides, out->ndim);
        int64_t ib = map_index(i, b, out->strides, out->ndim);
        out->data[i] = a->data[ia] + b->data[ib];
    }
}
