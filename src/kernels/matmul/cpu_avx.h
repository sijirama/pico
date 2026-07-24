#pragma once

#include <immintrin.h>
#include <pthread.h>
#include <stdbool.h>

#include "global.h"
#include "kernels/matmul/avx2_16x_exec.h"
#include "kernels/matmul/avx2_8x8_exec.h"
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

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_exec(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int row_start, int row_end, int columns,
    int k_dim) {
    pico_matmul_cpu_avx_16x_exec(a, b, out, row_start, row_end, columns, k_dim);
}

struct ThreadArgs {
    struct PicoTensor* a;
    struct PicoTensor* b;
    struct PicoTensor* out;

    int row_start;
    int row_end;

    int columns;
    int k_dim;
};

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_thread_entry(void* arg) {
    struct ThreadArgs* thread_args = (struct ThreadArgs*)arg;
    pico_matmul_cpu_avx_exec(thread_args->a, thread_args->b, thread_args->out, thread_args->row_start,
                             thread_args->row_end, thread_args->columns, thread_args->k_dim);
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx(struct PicoTensor* a, struct PicoTensor* b,
                                                                           struct PicoTensor* out) {
    int k_dim = a->shape[1];
    int columns = b->shape[1];
    int rows = a->shape[0];

    int i = 0;
    pico_matmul_cpu_avx_16x_exec(a, b, out, i, rows, columns, k_dim);
}
