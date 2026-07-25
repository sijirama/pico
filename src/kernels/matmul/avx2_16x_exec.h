#pragma once

#include "kernels/matmul/avx2_6x16.h"
#include "tensor.h"

#ifndef MATMUL_OPENMP_MIN_FLOPS
#define MATMUL_OPENMP_MIN_FLOPS 16000000LL
#endif

#ifndef MATMUL_PREFETCH_B_DISTANCE
#define MATMUL_PREFETCH_B_DISTANCE 16
#endif

#ifndef MATMUL_CACHE_BLOCK_SIZE
#define MATMUL_CACHE_BLOCK_SIZE 64
#endif

// INFO: this file is the 16-column matmul family. the hot path tries to hit
// 6x16 tiles first, then smaller row/column tails. out is updated with += because
// cache blocking splits k into chunks, so each kk block only contributes part of
// the final dot product.

// INFO: this is a side ways prefetch for b panels. when the j loop is on
//  b[0..k_dim][j..j+15], the next j tile will use b[0..k_dim][j+16..j+31].
//  so before we call the 6x16 kernel for the current tile, we touch a few rows
//  from the next b panel and give the cpu a chance to pull them closer. this
//  stays outside the hot k loop, so we do less prefetch work while the fma loop
//  is running.
__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx16_prefetch_b_panel(
    struct PicoTensor* b, int k_dim, int columns, int j) {
    int prefetch_j = j + MATMUL_PREFETCH_B_DISTANCE;
    if(prefetch_j + 16 > columns)
        return;

    for(int k = 0; k < k_dim; k += 16) {
        __builtin_prefetch(&b->data[k * b->strides[0] + prefetch_j * b->strides[1]], 0, 3);
    }
}

static inline void pico_matmul_cpu_avx16_kernel_scalar_Xx1(struct PicoTensor* a, struct PicoTensor* b,
                                                           struct PicoTensor* out, int k_start, int k_end, int i, int j,
                                                           int roll) {
    float m_cells[roll];
    for(int k = k_start; k < k_end; k += 1) {
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
    int block = MATMUL_CACHE_BLOCK_SIZE;

    // INFO: ii/jj/kk are the cache blocks. inside each block, the microkernels
    // still work in their natural shapes, but they only walk a smaller slice of
    // rows, columns, and k before moving on.
    for(int ii = row_start; ii < row_end; ii += block) {
        int rows = MIN(ii + block, row_end);

        for(int jj = 0; jj < columns; jj += block) {
            int cols = MIN(jj + block, columns);

            for(int kk = 0; kk < k_dim; kk += block) {
                int k_end = MIN(kk + block, k_dim);
                int i = ii;
                int roll = 0;

                // INFO: always try the biggest row tile first. after 6-row chunks
                // are done, the same block falls through to 4, then 2, then 1.
                roll = 6;
                for(; i + roll <= rows; i += roll) {
                    int j = jj;
                    for(; j + 16 <= cols; j += 16) {
                        pico_matmul_cpu_avx16_prefetch_b_panel(b, k_end, cols, j);
                        pico_matmul_cpu_avx_kernel_6_16(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 8 <= cols; j += 8) {
                        pico_matmul_cpu_avx_kernel_6_8(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 4 <= cols; j += 4) {
                        pico_matmul_cpu_sse_kernel_6_4(a, b, out, kk, k_end, i, j);
                    }
                    for(; j < cols; j++) {
                        pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, kk, k_end, i, j, roll);
                    }
                }

                roll = 4;
                for(; i + roll <= rows; i += roll) {
                    int j = jj;
                    for(; j + 16 <= cols; j += 16) {
                        pico_matmul_cpu_avx_kernel_4_16(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 8 <= cols; j += 8) {
                        pico_matmul_cpu_avx16_kernel_4_8(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 4 <= cols; j += 4) {
                        pico_matmul_cpu_sse_kernel_4_4(a, b, out, kk, k_end, i, j);
                    }
                    for(; j < cols; j++) {
                        pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, kk, k_end, i, j, roll);
                    }
                }

                roll = 2;
                for(; i + roll <= rows; i += roll) {
                    int j = jj;
                    for(; j + 16 <= cols; j += 16) {
                        pico_matmul_cpu_avx_kernel_2_16(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 8 <= cols; j += 8) {
                        pico_matmul_cpu_avx16_kernel_2_8(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 4 <= cols; j += 4) {
                        pico_matmul_cpu_sse_kernel_2_4(a, b, out, kk, k_end, i, j);
                    }
                    for(; j < cols; j++) {
                        pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, kk, k_end, i, j, roll);
                    }
                }

                roll = 1;
                for(; i + roll <= rows; i += roll) {
                    int j = jj;
                    for(; j + 16 <= cols; j += 16) {
                        pico_matmul_cpu_avx_kernel_1_16(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 8 <= cols; j += 8) {
                        pico_matmul_cpu_avx16_kernel_1_8(a, b, out, kk, k_end, i, j);
                    }
                    for(; j + 4 <= cols; j += 4) {
                        pico_matmul_cpu_sse_kernel_1_4(a, b, out, kk, k_end, i, j);
                    }
                    for(; j < cols; j++) {
                        pico_matmul_cpu_avx16_kernel_scalar_Xx1(a, b, out, kk, k_end, i, j, roll);
                    }
                }
            }
        }
    }
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx_16x(struct PicoTensor* a,
                                                                               struct PicoTensor* b,
                                                                               struct PicoTensor* out) {
    int rows = a->shape[0];
    int columns = b->shape[1];
    int k_dim = a->shape[1];
    int roll = 6;

    // NOTE: first row after the parallel 6-row tiles.
    // Example: rows=20, roll=6 -> parallel rows [0..17], tail starts at row 18.
    int tail_start = (rows / roll) * roll;
    long long flops = 2LL * (long long)rows * (long long)columns * (long long)k_dim;

    if(flops >= MATMUL_OPENMP_MIN_FLOPS) {
        // INFO: only full 6-row tiles go parallel. the tail stays serial so two
        // threads never write the same output rows.
        #pragma omp parallel for schedule(static)
        for(int i = 0; i <= rows - roll; i += roll) {
            pico_matmul_cpu_avx_16x_exec(a, b, out, i, i + roll, columns, k_dim);
        }

        pico_matmul_cpu_avx_16x_exec(a, b, out, tail_start, rows, columns, k_dim);
        return;
    }

    pico_matmul_cpu_avx_16x_exec(a, b, out, 0, rows, columns, k_dim);
}
