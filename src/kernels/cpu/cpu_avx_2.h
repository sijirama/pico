#pragma once
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include "tensor.h"

// AVX_2  element-wise add with broadcasting.
// AVX2 (Advanced Vector Extensions 2) is a SIMD (Single Instruction, Multiple Data) instruction set
// that allows a CPU to perform a single operation on multiple data elements simultaneously.
// Expanding on original AVX, AVX2 enables 256-bit wide vector processing for both floating-point
// numbers and integers

// WARN: plsssssss come back to this, i beg you @sijibomi
//
#define PICO_DEFINE_BINARY_OP_AVX2_FP32(name, simd_op, op)                    \
    __attribute__((target("avx2"))) static inline void name##_cpu_avx2_fp32(  \
        struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) { \
        int i = 0;                                                            \
        int size = out->numel;                                                \
        bool same_shape = pico_tensor_shapes_are_equal(a, b);                 \
        if(same_shape == true) {                                              \
            for(; i <= size - 8; i += 8) {                                    \
                __m256 va = _mm256_loadu_ps(&a->data[i]);                     \
                __m256 vb = _mm256_loadu_ps(&b->data[i]);                     \
                __m256 vres = simd_op(va, vb);                                \
                _mm256_storeu_ps(&out->data[i], vres);                        \
            }                                                                 \
            for(; i < size; i++) {                                            \
                out->data[i] = a->data[i] op b->data[i];                      \
            }                                                                 \
        } else {                                                              \
            for(int64_t i = 0; i < out->numel; i++) {                         \
                int64_t ia = map_index(i, a, out->strides, out->ndim);        \
                int64_t ib = map_index(i, b, out->strides, out->ndim);        \
                out->data[i] = a->data[ia] op b->data[ib];                    \
            }                                                                 \
        }                                                                     \
    }

PICO_DEFINE_BINARY_OP_AVX2_FP32(pico_add, _mm256_add_ps, +);
PICO_DEFINE_BINARY_OP_AVX2_FP32(pico_sub, _mm256_sub_ps, -);
PICO_DEFINE_BINARY_OP_AVX2_FP32(pico_mul, _mm256_mul_ps, *);
