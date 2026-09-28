#pragma once

#include "../../../global.h"
#include "scalar.h"
#include "cpu_avx.h"
#include "../../../tensor.h"

static inline void pico_matmul_cpu(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX2:
        case SIMD_AVX:
            pico_matmul_cpu_avx(a, b, out);
            break;
        default:
            pico_matmul_cpu_scalar(a, b, out);
    }
}

static inline void pico_grouped_matmul_cpu(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out,
                                           int group_size) {
    switch(g_simd_level) {
        case SIMD_AVX2:
        case SIMD_AVX:
            pico_grouped_matmul_cpu_avx(a, b, out, group_size);
            break;
        default:
            pico_grouped_matmul_cpu_scalar(a, b, out, group_size);
    }
}
