#pragma once
#include <immintrin.h>
#include <pthread.h>
#include <stdbool.h>

#include "global.h"
#include "tensor.h"
#include "tpool.h"

#ifndef MATMUL_THREAD_MAX
#define MATMUL_THREAD_MAX 8
#endif

#ifndef MATMUL_THREAD_MIN_ROWS
#define MATMUL_THREAD_MIN_ROWS 512
#endif

#ifndef MATMUL_THREAD_ROW_MAX
#define MATMUL_THREAD_ROW_MAX 64
#endif

static inline void pico_matmul_cpu_avx_kernel_scalar_Xx8(struct PicoTensor* a, struct PicoTensor* b,
                                                         struct PicoTensor* out, int k_dim, int i,
                                                         int j, int roll) {
    float m_cells[roll];
    for(int k = 0; k < k_dim; k += 1) {
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {
            m_cells[r] = a->data[(i + r) * a->strides[0] + k * a->strides[1]];  // M [i,K]
        }

        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {
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
    __attribute__((target("avx2,fma"), always_inline)) static inline void                          \
    pico_matmul_cpu_avx_kernel_##roll##_8(struct PicoTensor* a, struct PicoTensor* b,              \
                                          struct PicoTensor* out, int k_dim, int i, int j) {       \
        __m256 acc[roll];                                                                          \
                                                                                                   \
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                   \
            acc[r] = _mm256_loadu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]]); \
        }                                                                                          \
        for(int k = 0; k < k_dim; k++) {                                                           \
            __m256 m_vecs[roll];                                                                   \
            _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                               \
                m_vecs[r] = _mm256_set1_ps(a->data[(i + r) * a->strides[0] + k * a->strides[1]]);  \
            }                                                                                      \
            __m256 n_vec = _mm256_loadu_ps(&b->data[k * b->strides[0] + j * b->strides[1]]);       \
            _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                               \
                acc[r] = _mm256_fmadd_ps(m_vecs[r], n_vec, acc[r]);                                \
            }                                                                                      \
        }                                                                                          \
        _Pragma("GCC unroll 16") for(int r = 0; r < roll; r++) {                                   \
            _mm256_storeu_ps(&out->data[(i + r) * out->strides[0] + j * out->strides[1]], acc[r]); \
        }                                                                                          \
    }

PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(8);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(4);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(2);
PICO_DEFINE_MATMUL_CPU_AVX_MKERNEL_X(1);

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_exec(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int row_start, int row_end,
    int columns, int k_dim) {
    int i = row_start;
    int rows = row_end;
    int roll = 8;

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

struct ThreadArgs {
    struct PicoTensor* a;
    struct PicoTensor* b;
    struct PicoTensor* out;

    int row_start;  // inclusive
    int row_end;    // exclusive

    int columns;
    int k_dim;
};

__attribute__((target("avx2,fma"), always_inline)) static inline void
pico_matmul_cpu_avx_thread_entry(void* arg) {
    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    pico_matmul_cpu_avx_exec(thread_args->a, thread_args->b, thread_args->out,
                             thread_args->row_start, thread_args->row_end, thread_args->columns,
                             thread_args->k_dim);
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx(struct PicoTensor* a,
                                                                           struct PicoTensor* b,
                                                                           struct PicoTensor* out) {
    int k_dim = a->shape[1];
    int columns = b->shape[1];
    int rows = a->shape[0];

    if(rows < MATMUL_THREAD_MIN_ROWS) {
        int i = 0;
        pico_matmul_cpu_avx_exec(a, b, out, i, rows, columns, k_dim);
        return;
    }

    // INFO: multithreaded matmul

    // take the min btw our max threads and the allocatable
    // threads from the rows to avoid threads with no work
    // initialize the threads array with the proper count
    int row_chunks = (rows + MATMUL_THREAD_ROW_MAX - 1) / MATMUL_THREAD_ROW_MAX;
    int thread_count = MIN(MATMUL_THREAD_MAX, row_chunks);
    struct ThreadArgs* args = (struct ThreadArgs*)malloc(thread_count * sizeof(struct ThreadArgs));

    int rows_per_thread = rows / thread_count;
    int row_tail = rows % thread_count;

    int current_row = 0;

    for(int thread = 0; thread < thread_count; thread++) {
        int start_row = current_row;

        int rows_this_thread = rows_per_thread;

        if(thread == thread_count - 1) {
            rows_this_thread += row_tail;
        }

        int end_row = start_row + rows_this_thread;

        args[thread].a = a;
        args[thread].b = b;
        args[thread].out = out;
        args[thread].row_start = start_row;
        args[thread].row_end = end_row;
        args[thread].columns = columns;
        args[thread].k_dim = k_dim;

        pico_tpool_add_work(global_tp, pico_matmul_cpu_avx_thread_entry, &args[thread]);

        current_row = end_row;
    }

    pico_tpool_wait(global_tp);

    free(args);
}
