#pragma once

#include <immintrin.h>

#include "tensor.h"

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_kernel_6_16(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int k_dim, int i, int j) {
    __m256 b_vec_0;
    __m256 b_vec_1;
    __m256 m_vec_a;
    __m256 m_vec_b;

    // 6 rows of C, 16 columns split across two AVX registers per row.
    __m256 acc0_0 = _mm256_loadu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc0_1 = _mm256_loadu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1]) + 8]);
    __m256 acc1_0 = _mm256_loadu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc1_1 = _mm256_loadu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1]) + 8]);
    __m256 acc2_0 = _mm256_loadu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc2_1 = _mm256_loadu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1]) + 8]);
    __m256 acc3_0 = _mm256_loadu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc3_1 = _mm256_loadu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1]) + 8]);
    __m256 acc4_0 = _mm256_loadu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc4_1 = _mm256_loadu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1]) + 8]);
    __m256 acc5_0 = _mm256_loadu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1]) + 0]);
    __m256 acc5_1 = _mm256_loadu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1]) + 8]);

    for(int k = 0; k < k_dim; k++) {
        b_vec_0 = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1] + 0]);
        b_vec_1 = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1] + 8]);

        m_vec_a = _mm256_set1_ps(a->data[(i + 0) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 1) * a->strides[0] + k * a->strides[1]]);

        acc0_0 = _mm256_fmadd_ps(b_vec_0, m_vec_a, acc0_0);
        acc0_1 = _mm256_fmadd_ps(b_vec_1, m_vec_a, acc0_1);
        acc1_0 = _mm256_fmadd_ps(b_vec_0, m_vec_b, acc1_0);
        acc1_1 = _mm256_fmadd_ps(b_vec_1, m_vec_b, acc1_1);

        m_vec_a = _mm256_set1_ps(a->data[(i + 2) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 3) * a->strides[0] + k * a->strides[1]]);

        acc2_0 = _mm256_fmadd_ps(b_vec_0, m_vec_a, acc2_0);
        acc2_1 = _mm256_fmadd_ps(b_vec_1, m_vec_a, acc2_1);
        acc3_0 = _mm256_fmadd_ps(b_vec_0, m_vec_b, acc3_0);
        acc3_1 = _mm256_fmadd_ps(b_vec_1, m_vec_b, acc3_1);

        m_vec_a = _mm256_set1_ps(a->data[(i + 4) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 5) * a->strides[0] + k * a->strides[1]]);

        acc4_0 = _mm256_fmadd_ps(b_vec_0, m_vec_a, acc4_0);
        acc4_1 = _mm256_fmadd_ps(b_vec_1, m_vec_a, acc4_1);
        acc5_0 = _mm256_fmadd_ps(b_vec_0, m_vec_b, acc5_0);
        acc5_1 = _mm256_fmadd_ps(b_vec_1, m_vec_b, acc5_1);
    }

    _mm256_storeu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1]) + 0], acc0_0);
    _mm256_storeu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1]) + 8], acc0_1);
    _mm256_storeu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1]) + 0], acc1_0);
    _mm256_storeu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1]) + 8], acc1_1);
    _mm256_storeu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1]) + 0], acc2_0);
    _mm256_storeu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1]) + 8], acc2_1);
    _mm256_storeu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1]) + 0], acc3_0);
    _mm256_storeu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1]) + 8], acc3_1);
    _mm256_storeu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1]) + 0], acc4_0);
    _mm256_storeu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1]) + 8], acc4_1);
    _mm256_storeu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1]) + 0], acc5_0);
    _mm256_storeu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1]) + 8], acc5_1);
}

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_kernel_6_8(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int k_dim, int i, int j) {
    __m256 b_vec;
    __m256 m_vec_a;
    __m256 m_vec_b;

    __m256 acc0 = _mm256_loadu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1])]);
    __m256 acc1 = _mm256_loadu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1])]);
    __m256 acc2 = _mm256_loadu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1])]);
    __m256 acc3 = _mm256_loadu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1])]);
    __m256 acc4 = _mm256_loadu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1])]);
    __m256 acc5 = _mm256_loadu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1])]);

    for(int k = 0; k < k_dim; k++) {
        b_vec = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1]]);

        m_vec_a = _mm256_set1_ps(a->data[(i + 0) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 1) * a->strides[0] + k * a->strides[1]]);

        acc0 = _mm256_fmadd_ps(b_vec, m_vec_a, acc0);
        acc1 = _mm256_fmadd_ps(b_vec, m_vec_b, acc1);

        m_vec_a = _mm256_set1_ps(a->data[(i + 2) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 3) * a->strides[0] + k * a->strides[1]]);

        acc2 = _mm256_fmadd_ps(b_vec, m_vec_a, acc2);
        acc3 = _mm256_fmadd_ps(b_vec, m_vec_b, acc3);

        m_vec_a = _mm256_set1_ps(a->data[(i + 4) * a->strides[0] + k * a->strides[1]]);
        m_vec_b = _mm256_set1_ps(a->data[(i + 5) * a->strides[0] + k * a->strides[1]]);

        acc4 = _mm256_fmadd_ps(b_vec, m_vec_a, acc4);
        acc5 = _mm256_fmadd_ps(b_vec, m_vec_b, acc5);
    }

    _mm256_storeu_ps(&out->data[(i + 0) * out->strides[0] + (j * out->strides[1])], acc0);
    _mm256_storeu_ps(&out->data[(i + 1) * out->strides[0] + (j * out->strides[1])], acc1);
    _mm256_storeu_ps(&out->data[(i + 2) * out->strides[0] + (j * out->strides[1])], acc2);
    _mm256_storeu_ps(&out->data[(i + 3) * out->strides[0] + (j * out->strides[1])], acc3);
    _mm256_storeu_ps(&out->data[(i + 4) * out->strides[0] + (j * out->strides[1])], acc4);
    _mm256_storeu_ps(&out->data[(i + 5) * out->strides[0] + (j * out->strides[1])], acc5);
}
