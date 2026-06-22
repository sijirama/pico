#pragma once

#include "global.h"
#include "tensor.h"
#include "kernels/cpu/scalar.h"

// CPU dispatch: pick the kernel variant for the detected SIMD level.
// g_simd_level is set once by pico_init(); default falls back to scalar so an
// unknown/unsupported level still computes correctly (just slower).
static inline void pico_add_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_add_cpu_scalar(a, b, out);
    }
}

static inline void pico_sub_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_sub_cpu_scalar(a, b, out);
    }
}

static inline void pico_mul_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_mul_cpu_scalar(a, b, out);
    }
}
