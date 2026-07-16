#pragma once
#include <immintrin.h>
#include <stdbool.h>

#include "tensor.h"

/*
 * ok so we want to loop right and run our normal ikj setup but this time we wanna use avx intinsics
 *
 *
 * */

__attribute__((target("avx"))) static inline void pico_matmul_cpu_avx(struct PicoTensor* a,
                                                                      struct PicoTensor* b,
                                                                      struct PicoTensor* out) {
    int rows = a->shape[0];
    int columns = b->shape[1];
    int k_dim = a->shape[1];

    for(int i = 0; i < rows; i++) {
        for(int k = 0; k < k_dim; k++) {
            float m_cell = a->data[i * a->strides[0] + k * a->strides[1]];

            int j = 0;

            for(; j + 8 <= columns; j += 8) {
                __m256 m_vec = _mm256_set1_ps(m_cell);

                __m256 n_vec = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1]]);
                __m256 o_vec =
                    _mm256_loadu_ps(&out->data[i * out->strides[0] + j * out->strides[1]]);

                __m256 res_vec = _mm256_mul_ps(m_vec, n_vec);

                _mm256_store_ps(&out->data[i * out->strides[0] + j * out->strides[1]],
                                _mm256_add_ps(o_vec, res_vec));
            }

            for(; j < columns; j++) {
                out->data[i * out->strides[0] + j * out->strides[1]] +=
                    m_cell * b->data[k * b->strides[0] + j * b->strides[1]];
            }
        }
    }
}

void pico_matmul_cpu_avx_kernel_2_8(struct PicoTensor* a, struct PicoTensor* b,
                                    struct PicoTensor* out, int k_dim, int columns, int i) {}
