/*
 * matmul benchmark: scalar vs AVX kernel.
 *
 * Build/run with `make bench` (compiles at -O2 — benchmarking a -g/-O0 build is
 * meaningless). We call the raw kernels directly (pico_matmul_cpu_scalar /
 * pico_matmul_cpu_avx), not the g_simd_level wrapper, so this measures the
 * kernels themselves with no dispatch in the way.
 *
 * CAVEAT to read the number honestly: at -O2 the "scalar" loop may get
 * auto-vectorized to SSE (4-wide) by the compiler, while our hand kernel is
 * AVX (8-wide). So the speedup is "our AVX vs an already-optimized scalar
 * baseline", not a raw 8x. That gap IS the thing worth understanding.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

#define N 512      // square matrices N x N
#define WARMUP 3   // untimed runs to warm caches / branch predictors
#define ITERS 20   // timed runs, averaged

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef void (*matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

// avg seconds per matmul. zero `out` each iter because matmul accumulates (+=).
static double bench_kernel(matmul_fn fn, struct PicoTensor* a, struct PicoTensor* b,
                           struct PicoTensor* out, int iters) {
    size_t bytes = (size_t)out->numel * sizeof(float);
    for(int w = 0; w < WARMUP; w++) {
        memset(out->data, 0, bytes);
        fn(a, b, out);
    }
    double t0 = now_sec();
    for(int it = 0; it < iters; it++) {
        memset(out->data, 0, bytes);
        fn(a, b, out);
    }
    double t1 = now_sec();
    return (t1 - t0) / (double)iters;
}

int main(void) {
    pico_init();

    int64_t shape[] = {N, N};
    struct PicoTensor* a = pico_param(shape, 2);
    struct PicoTensor* b = pico_param(shape, 2);
    struct PicoTensor* out = pico_param(shape, 2);
    struct PicoTensor* ref = pico_param(shape, 2);

    // deterministic small values (kept small so the accumulation stays exact-ish)
    for(int64_t i = 0; i < a->numel; i++) {
        a->data[i] = (float)((i % 13) - 6) * 0.25f;
        b->data[i] = (float)((i % 7) - 3) * 0.5f;
    }

    // correctness gate: AVX must match scalar before the timing means anything
    size_t bytes = (size_t)out->numel * sizeof(float);
    memset(ref->data, 0, bytes);
    pico_matmul_cpu_scalar(a, b, ref);
    memset(out->data, 0, bytes);
    pico_matmul_cpu_avx(a, b, out);

    float max_diff = 0.0f;
    for(int64_t i = 0; i < out->numel; i++) {
        float d = fabsf(out->data[i] - ref->data[i]);
        if(d > max_diff) max_diff = d;
    }
    int correct = (max_diff <= 1e-3f);

    double t_scalar = bench_kernel(pico_matmul_cpu_scalar, a, b, out, ITERS);
    double t_avx = bench_kernel(pico_matmul_cpu_avx, a, b, out, ITERS);

    double flops = 2.0 * (double)N * (double)N * (double)N;  // per matmul
    double g_scalar = flops / t_scalar / 1e9;
    double g_avx = flops / t_avx / 1e9;

    printf("\n  pico matmul benchmark   (N=%d, iters=%d, -O2)\n", N, ITERS);
    printf("  ---------------------------------------------\n");
    printf("  correctness: max|avx - scalar| = %.3e  [%s]\n\n", max_diff,
           correct ? "OK" : "MISMATCH -- numbers below are meaningless");
    printf("  scalar : %8.3f ms/matmul   %6.2f GFLOP/s\n", t_scalar * 1e3, g_scalar);
    printf("  avx    : %8.3f ms/matmul   %6.2f GFLOP/s\n", t_avx * 1e3, g_avx);
    printf("  ---------------------------------------------\n");
    printf("  speedup (scalar/avx): %.2fx\n\n", t_scalar / t_avx);

    pico_free(a);
    pico_free(b);
    pico_free(out);
    pico_free(ref);
    return correct ? 0 : 1;
}
