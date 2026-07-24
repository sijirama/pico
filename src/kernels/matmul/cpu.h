#pragma once

#include "global.h"
#include "kernels/matmul/cpu_avx.h"
#include "kernels/matmul/scalar.h"
#include "tensor.h"

static inline void pico_matmul_cpu(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX:
            pico_matmul_cpu_avx(a, b, out);
            break;
        default:
            pico_matmul_cpu_scalar(a, b, out);
    }
}
