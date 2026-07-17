/*
 * bench_common.h — shared benchmark utilities (bench-only, never shipped).
 *
 * Timing, a scalar reference cell, and per-roll FULL-matmul drivers built on top
 * of the AVX microkernels (pico_matmul_cpu_avx_kernel_{1,2,4,8}_8). Each driver
 * tiles the whole matrix at ONE fixed roll width (R rows x 8 cols) and falls back
 * to scalar for the row/col tails — so we can measure each microkernel width as a
 * standalone strategy and watch register pressure decide the winner.
 *
 * out MUST be zeroed by the caller before every call (the kernels accumulate, +=).
 */
#pragma once

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "kernels/cpu_kernels.h"  // scalar + AVX microkernels (all static inline)
#include "tensor.h"

static inline double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// one output cell, full reduction over k. used for tail regions + as a building
// block. accumulates into out (out[i][j] += sum_k a[i][k]*b[k][j]).
static inline void bench_scalar_cell(struct PicoTensor* a, struct PicoTensor* b,
                                     struct PicoTensor* out, int k_dim, int i, int j) {
    float acc = 0.0f;
    for(int k = 0; k < k_dim; k++)
        acc += a->data[i * a->strides[0] + k * a->strides[1]] *
               b->data[k * b->strides[0] + j * b->strides[1]];
    out->data[i * out->strides[0] + j * out->strides[1]] += acc;
}

// stamp a full-matmul driver that uses ONLY the R-row microkernel for R x 8 tiles,
// scalar for the right strip (R rows x tail cols) and bottom strip (tail rows).
#define BENCH_DEFINE_ROLL_DRIVER(R)                                                    \
    __attribute__((target("avx2,fma"))) static inline void bench_matmul_roll##R(       \
        struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {           \
        int rows = a->shape[0];                                                        \
        int cols = b->shape[1];                                                        \
        int k_dim = a->shape[1];                                                       \
        int i = 0;                                                                     \
        for(; i + (R) <= rows; i += (R)) {                                             \
            int j = 0;                                                                 \
            for(; j + 8 <= cols; j += 8)                                               \
                pico_matmul_cpu_avx_kernel_##R##_8(a, b, out, k_dim, i, j);            \
            for(; j < cols; j++) /* right strip: R rows x 1 col */                     \
                for(int r = 0; r < (R); r++)                                           \
                    bench_scalar_cell(a, b, out, k_dim, i + r, j);                     \
        }                                                                              \
        for(; i < rows; i++) /* bottom strip: leftover rows, all cols */               \
            for(int j = 0; j < cols; j++)                                              \
                bench_scalar_cell(a, b, out, k_dim, i, j);                             \
    }

BENCH_DEFINE_ROLL_DRIVER(1)
BENCH_DEFINE_ROLL_DRIVER(2)
BENCH_DEFINE_ROLL_DRIVER(4)
BENCH_DEFINE_ROLL_DRIVER(8)
BENCH_DEFINE_ROLL_DRIVER(14)

// signature shared by every matmul strategy (scalar, roll drivers, adaptive avx)
typedef void (*bench_matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

// avg seconds per matmul over `iters` timed runs (after `warmup`), zeroing out each time.
static inline double bench_time_matmul(bench_matmul_fn fn, struct PicoTensor* a,
                                       struct PicoTensor* b, struct PicoTensor* out, int warmup,
                                       int iters) {
    size_t bytes = (size_t)out->numel * sizeof(float);
    for(int w = 0; w < warmup; w++) {
        memset(out->data, 0, bytes);
        fn(a, b, out);
    }
    double t0 = bench_now_sec();
    for(int it = 0; it < iters; it++) {
        memset(out->data, 0, bytes);
        fn(a, b, out);
    }
    return (bench_now_sec() - t0) / (double)iters;
}

// max abs difference between two same-shape tensors (correctness gate helper)
static inline float bench_max_abs_diff(struct PicoTensor* x, struct PicoTensor* y) {
    float m = 0.0f;
    for(int64_t i = 0; i < x->numel; i++) {
        float d = fabsf(x->data[i] - y->data[i]);
        if(d > m) m = d;
    }
    return m;
}
