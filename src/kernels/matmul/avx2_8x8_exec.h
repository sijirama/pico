#pragma once

#include "kernels/matmul/avx2_8x8.h"
#include "tensor.h"

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_8x8_exec(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int row_start, int row_end, int columns,
    int k_dim) {
    int i = row_start;
    int rows = row_end;
    int roll = 0;

    roll = 8;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_8_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 4;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_4_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 2;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_2_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_Xx8(a, b, out, k_dim, i, j, roll);
        }
    }

    roll = 1;
    for(; i + roll <= rows; i += roll) {
        int j = 0;
        for(; j + 8 <= columns; j += 8) {
            pico_matmul_cpu_avx_kernel_1_8(a, b, out, k_dim, i, j);
        }
        for(; j < columns; j++) {
            pico_matmul_cpu_avx_kernel_scalar_1x8(a, b, out, k_dim, i, j);
        }
    }
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx_8x8(struct PicoTensor* a,
                                                                               struct PicoTensor* b,
                                                                               struct PicoTensor* out) {
    pico_matmul_cpu_avx_8x8_exec(a, b, out, 0, a->shape[0], b->shape[1], a->shape[1]);
}
