#pragma once
#include <immintrin.h>
#include <stdbool.h>

#include "tensor.h"

static inline void pico_matmul_cpu_avx_kernel_scalar_Xx8(struct PicoTensor* a, struct PicoTensor* b,
                                                         struct PicoTensor* out, int k_dim, int i,
                                                         int j, int roll) {
    float m_cells[roll];
    for(int k = 0; k < k_dim; k += 1) {
        for(int r = 0; r < roll; r++) {
            m_cells[r] = a->data[(i + r) * a->strides[0] + k * a->strides[1]];  // M [i,K]
        }

        for(int r = 0; r < roll; r++) {
            out->data[(i + r) * out->strides[0] + j * out->strides[1]] +=
                m_cells[r] * b->data[k * b->strides[0] + j * b->strides[1]];
        }
    }
}

static inline void pico_matmul_cpu_avx_kernel_scalar_1x8(struct PicoTensor* a, struct PicoTensor* b,
                                                         struct PicoTensor* out, int k_dim, int i,
                                                         int j) {
    for(int k = 0; k < k_dim; k++) {
        float m_cell = a->data[i * a->strides[0] + k * a->strides[1]];  // M [i,K]

        out->data[i * out->strides[0] + j * out->strides[1]] +=
            m_cell * b->data[k * b->strides[0] + j * b->strides[1]];
    }
}

#define PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(roll)                                                 \
    __attribute__((target("avx"))) static inline void pico_matmul_cpu_avx_kernel_##roll##_8(        \
        struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int k_dim, int i,      \
        int j) {                                                                                   \
        __m256 acc[roll];                                                                          \
                                                                                                   \
        for(int r = 0; r < roll; r++) {                                                            \
            acc[r] = _mm256_loadu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]]); \
        }                                                                                          \
        for(int k = 0; k < k_dim; k++) {                                                           \
            __m256 m_vecs[roll];                                                                   \
            for(int r = 0; r < roll; r++) {                                                        \
                m_vecs[r] = _mm256_set1_ps(a->data[(i + r) * a->strides[0] + k * a->strides[1]]);  \
            }                                                                                      \
            __m256 n_vec = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1]]);       \
            for(int r = 0; r < roll; r++) {                                                        \
                acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(m_vecs[r], n_vec));                   \
            }                                                                                      \
        }                                                                                          \
        for(int r = 0; r < roll; r++) {                                                            \
            _mm256_storeu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]], acc[r]); \
        }                                                                                          \
    }

PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(8);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(4);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(2);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(1);

static inline void pico_matmul_cpu_avx(struct PicoTensor* a, struct PicoTensor* b,
                                       struct PicoTensor* out) {
    int rows = a->shape[0];
    int columns = b->shape[1];
    int k_dim = a->shape[1];

    int i = 0;

    for(; i + 8 <= rows; i += 8) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_8_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, 8);
        }
    }

    for(; i + 4 <= rows; i += 4) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_4_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, 4);
        }
    }

    for(; i + 2 <= rows; i += 2) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_2_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, 2);
        }
    }

    for(; i + 1 <= rows; i += 1) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_1_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_1x8(a, b, out, k_dim, i, j);
        }
    }
}
