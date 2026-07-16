#pragma once

#include "global.h"
#include "kernels/cpu/cpu_avx.h"
#include "kernels/cpu/cpu_avx_2.h"
#include "kernels/cpu/cpu_scalar.h"
#include "tensor.h"

// CPU dispatch: pick the kernel variant for the detected SIMD level.
// g_simd_level is set once by pico_init(); default falls back to scalar so an
// unknown/unsupported level still computes correctly (just slower).

static inline void pico_add_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX2:
            pico_add_cpu_avx2_fp32(a, b, out);
            break;
        default:
            pico_add_cpu_scalar(a, b, out);
    }
}

static inline void pico_sub_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX2:
            pico_sub_cpu_avx2_fp32(a, b, out);
            break;
        default:
            pico_sub_cpu_scalar(a, b, out);
    }
}

static inline void pico_mul_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX2:
            pico_mul_cpu_avx2_fp32(a, b, out);
            break;
        default:
            pico_mul_cpu_scalar(a, b, out);
    }
}

static inline void pico_matmul_cpu(struct PicoTensor* a, struct PicoTensor* b,
                                   struct PicoTensor* out) {
    switch(g_simd_level) {
        case SIMD_AVX:
            pico_matmul_cpu_avx(a, b, out);
            break;
        default:
            pico_matmul_cpu_scalar(a, b, out);
    }
}

// unary element-wise math. same switch shape as above — ripe for the future
// pico_op_cpu(a, out, op) bundling once there's more than one SIMD variant.

static inline void pico_sqrt_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_sqrt_cpu_scalar(a, out);
    }
}

static inline void pico_sin_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_sin_cpu_scalar(a, out);
    }
}

static inline void pico_cos_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_cos_cpu_scalar(a, out);
    }
}

static inline void pico_tan_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_tan_cpu_scalar(a, out);
    }
}

static inline void pico_tanh_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_tanh_cpu_scalar(a, out);
    }
}

static inline void pico_log_cpu(struct PicoTensor* a, struct PicoTensor* out) {
    switch(g_simd_level) {
        default:
            pico_log_cpu_scalar(a, out);
    }
}
