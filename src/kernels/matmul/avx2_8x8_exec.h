#pragma once

#include "kernels/matmul/avx2_8x8.h"
#include "tensor.h"

#ifndef MATMUL_OPENMP_MIN_FLOPS
#define MATMUL_OPENMP_MIN_FLOPS 16000000LL
#endif

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
    int rows = a->shape[0];
    int columns = b->shape[1];
    int k_dim = a->shape[1];
    int roll = 8;

    // NOTE: first row not covered by the parallel 8-row tiles.
    // Example: rows=20, roll=8 -> parallel rows [0..15], tail starts at row 16.
    int tail_start = (rows / roll) * roll;
    long long flops = 2LL * (long long)rows * (long long)columns * (long long)k_dim;

    if(flops >= MATMUL_OPENMP_MIN_FLOPS) {
#pragma omp parallel for schedule(static)
        for(int i = 0; i <= rows - roll; i += roll) {
            pico_matmul_cpu_avx_8x8_exec(a, b, out, i, i + roll, columns, k_dim);
        }

        pico_matmul_cpu_avx_8x8_exec(a, b, out, tail_start, rows, columns, k_dim);
        return;
    }

    pico_matmul_cpu_avx_8x8_exec(a, b, out, 0, rows, columns, k_dim);
}
