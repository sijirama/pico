#pragma once

#include <immintrin.h>

#include "tensor.h"

static inline void pico_matmul_cpu_avx_kernel_scalar_Xx8(struct PicoTensor* a, struct PicoTensor* b,
                                                         struct PicoTensor* out, int k_dim, int i, int j, int roll) {
    float m_cells[roll];
    for(int k = 0; k < k_dim; k += 1) {
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {
            m_cells[r] = a->data[(i + r) * a->strides[0] + k * a->strides[1]];
        }

        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {
            out->data[(i + r) * out->strides[0] + j * out->strides[1]] +=
                m_cells[r] * b->data[k * b->strides[0] + j * b->strides[1]];
        }
    }
}

static inline void pico_matmul_cpu_avx_kernel_scalar_1x8(struct PicoTensor* a, struct PicoTensor* b,
                                                         struct PicoTensor* out, int k_dim, int i, int j) {
    for(int k = 0; k < k_dim; k++) {
        float m_cell = a->data[i * a->strides[0] + k * a->strides[1]];

        out->data[i * out->strides[0] + j * out->strides[1]] += m_cell * b->data[k * b->strides[0] + j * b->strides[1]];
    }
}

#define PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(roll)                                                               \
    __attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_kernel_##roll##_8( \
        struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int k_dim, int i, int j) {           \
        __m256 acc[roll];                                                                                        \
                                                                                                                 \
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                                 \
            acc[r] = _mm256_loadu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]]);               \
        }                                                                                                        \
                                                                                                                 \
        for(int k = 0; k < k_dim; k++) {                                                                         \
            __m256 m_vecs[roll];                                                                                 \
            _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                             \
                m_vecs[r] = _mm256_set1_ps(a->data[(i + r) * a->strides[0] + k * a->strides[1]]);                \
            }                                                                                                    \
                                                                                                                 \
            __m256 n_vec = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1]]);                     \
                                                                                                                 \
            _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                             \
                acc[r] = _mm256_fmadd_ps(m_vecs[r], n_vec, acc[r]);                                              \
            }                                                                                                    \
        }                                                                                                        \
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                                 \
            _mm256_storeu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]], acc[r]);               \
        }                                                                                                        \
    }

PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(8);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(4);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(2);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(1);
