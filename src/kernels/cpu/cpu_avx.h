#pragma once
#include <immintrin.h>
#include <pthread.h>
#include <stdbool.h>

#include "tensor.h"

#define MATMUL_THREAD_MAX 8
#define MATMUL_THREAD_MIN_ROWS 512
#define MATMUL_THREAD_ROW_MAX 64

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

#define PICO_MATMUL_CPU_AVX_EXEC(a_arg, b_arg, out_arg, row_start_arg, row_end_arg, columns_arg, \
                                 k_dim_arg)                                                      \
    {                                                                                            \
        int mm_i = (row_start_arg);                                                              \
        int mm_rows = (row_end_arg);                                                             \
        int mm_roll = 8;                                                                         \
        for(; mm_i + mm_roll <= mm_rows; mm_i += mm_roll) {                                      \
            int mm_j = 0;                                                                        \
            for(; mm_j + 8 <= (columns_arg); mm_j += 8) {                                        \
                pico_matmul_cpu_avx_kernel_8_8((a_arg), (b_arg), (out_arg), (k_dim_arg), mm_i,   \
                                               mm_j);                                            \
            }                                                                                    \
            for(; mm_j < (columns_arg); mm_j++) {                                                \
                pico_matmul_cpu_avx_kernel_scalar_Xx8((a_arg), (b_arg), (out_arg), (k_dim_arg),  \
                                                      mm_i, mm_j, mm_roll);                      \
            }                                                                                    \
        }                                                                                        \
        mm_roll = 4;                                                                             \
        for(; mm_i + mm_roll <= mm_rows; mm_i += mm_roll) {                                      \
            int mm_j = 0;                                                                        \
            for(; mm_j + 8 <= (columns_arg); mm_j += 8) {                                        \
                pico_matmul_cpu_avx_kernel_4_8((a_arg), (b_arg), (out_arg), (k_dim_arg), mm_i,   \
                                               mm_j);                                            \
            }                                                                                    \
            for(; mm_j < (columns_arg); mm_j++) {                                                \
                pico_matmul_cpu_avx_kernel_scalar_Xx8((a_arg), (b_arg), (out_arg), (k_dim_arg),  \
                                                      mm_i, mm_j, mm_roll);                      \
            }                                                                                    \
        }                                                                                        \
        mm_roll = 2;                                                                             \
        for(; mm_i + mm_roll <= mm_rows; mm_i += mm_roll) {                                      \
            int mm_j = 0;                                                                        \
            for(; mm_j + 8 <= (columns_arg); mm_j += 8) {                                        \
                pico_matmul_cpu_avx_kernel_2_8((a_arg), (b_arg), (out_arg), (k_dim_arg), mm_i,   \
                                               mm_j);                                            \
            }                                                                                    \
            for(; mm_j < (columns_arg); mm_j++) {                                                \
                pico_matmul_cpu_avx_kernel_scalar_Xx8((a_arg), (b_arg), (out_arg), (k_dim_arg),  \
                                                      mm_i, mm_j, mm_roll);                      \
            }                                                                                    \
        }                                                                                        \
        mm_roll = 1;                                                                             \
        for(; mm_i + mm_roll <= mm_rows; mm_i += mm_roll) {                                      \
            int mm_j = 0;                                                                        \
            for(; mm_j + 8 <= (columns_arg); mm_j += 8) {                                        \
                pico_matmul_cpu_avx_kernel_1_8((a_arg), (b_arg), (out_arg), (k_dim_arg), mm_i,   \
                                               mm_j);                                            \
            }                                                                                    \
            for(; mm_j < (columns_arg); mm_j++) {                                                \
                pico_matmul_cpu_avx_kernel_scalar_1x8((a_arg), (b_arg), (out_arg), (k_dim_arg),  \
                                                      mm_i, mm_j);                               \
            }                                                                                    \
        }                                                                                        \
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

__attribute__((target("avx2,fma"), always_inline)) static inline void*
pico_matmul_cpu_avx_thread_entry(void* arg) {
    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    PICO_MATMUL_CPU_AVX_EXEC(thread_args->a, thread_args->b, thread_args->out,
                             thread_args->row_start, thread_args->row_end, thread_args->columns,
                             thread_args->k_dim);
    return NULL;
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx(struct PicoTensor* a,
                                                                           struct PicoTensor* b,
                                                                           struct PicoTensor* out) {
    int k_dim = a->shape[1];
    int columns = b->shape[1];
    int rows = a->shape[0];

    if(rows < MATMUL_THREAD_MIN_ROWS) {
        int i = 0;
        PICO_MATMUL_CPU_AVX_EXEC(a, b, out, i, rows, columns, k_dim);
        return;
    }

    // INFO: multithreaded matmul

    // take the min btw our max threads and the allocatable
    // threads from the rows to avoid threads with no work
    // initialize the threads array with the proper count
    pthread_t* threads;
    int row_chunks = (rows + MATMUL_THREAD_ROW_MAX - 1) / MATMUL_THREAD_ROW_MAX;
    int thread_count = MIN(MATMUL_THREAD_MAX, row_chunks);
    threads = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
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

        pthread_create(&threads[thread], NULL, pico_matmul_cpu_avx_thread_entry, &args[thread]);

        current_row = end_row;
    }

    for(int thread = 0; thread < thread_count; thread++) {
        pthread_join(threads[thread], NULL);
    }

    free(args);
    free(threads);
}
