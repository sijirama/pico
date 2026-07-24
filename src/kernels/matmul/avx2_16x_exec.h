#pragma once

#include "kernels/matmul/avx2_6x16.h"
#include "tensor.h"

static inline void pico_matmul_cpu_avx16_kernel_scalar_Xx1(struct PicoTensor* a, struct PicoTensor* b,
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

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_16x_exec(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int row_start, int row_end, int columns,
    int k_dim) {
    int i = row_start;
    int rows = row_end;
    int roll = 0;

    roll = 6;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 16 <= columns; j += 16) {
            pico_matmul_cpu_avx_kernel_6_16(a, b, out, k_dim, i, j);
        }
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_6_8(a, b, out, k_dim, i, j);
        }
        for(; j + 4 <= columns; j += 4) {
            pico_matmul_cpu_sse_kernel_6_4(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 4;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 16 <= columns; j += 16) {
            pico_matmul_cpu_avx_kernel_4_16(a, b, out, k_dim, i, j);
        }
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx16_kernel_4_8(a, b, out, k_dim, i, j);
        }
        for(; j + 4 <= columns; j += 4) {
            pico_matmul_cpu_sse_kernel_4_4(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 2;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 16 <= columns; j += 16) {
            pico_matmul_cpu_avx_kernel_2_16(a, b, out, k_dim, i, j);
        }
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx16_kernel_2_8(a, b, out, k_dim, i, j);
        }
        for(; j + 4 <= columns; j += 4) {
            pico_matmul_cpu_sse_kernel_2_4(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 1;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 16 <= columns; j += 16) {
            pico_matmul_cpu_avx_kernel_1_16(a, b, out, k_dim, i, j);
        }
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx16_kernel_1_8(a, b, out, k_dim, i, j);
        }
        for(; j + 4 <= columns; j += 4) {
            pico_matmul_cpu_sse_kernel_1_4(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, k_dim, i, j, roll);
        }
    }
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx_16x(struct PicoTensor* a,
                                                                               struct PicoTensor* b,
                                                                               struct PicoTensor* out) {
    pico_matmul_cpu_avx_16x_exec(a, b, out, 0, a->shape[0], b->shape[1], a->shape[1]);
}
