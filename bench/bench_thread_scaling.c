/*
 * thread_scaling benchmark: compare matmul behavior across larger square sizes.
 *
 * Run with `make thread_scaling` from bench/.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

#define WARMUP 2
#define SAMPLES 5
#define TOL 1e-3f

typedef void (*matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median_time(matmul_fn fn, struct PicoTensor* a, struct PicoTensor* b,
                          struct PicoTensor* out) {
    double samples[SAMPLES];
    size_t bytes = (size_t)out->numel * sizeof(float);

    for(int w = 0; w < WARMUP; w++) {
        memset(out->data, 0, bytes);
        fn(a, b, out);
    }

    for(int s = 0; s < SAMPLES; s++) {
        memset(out->data, 0, bytes);
        double t0 = now_sec();
        fn(a, b, out);
        samples[s] = now_sec() - t0;
    }

    qsort(samples, SAMPLES, sizeof(double), cmp_double);
    return samples[SAMPLES / 2];
}

static float max_abs_diff(struct PicoTensor* x, struct PicoTensor* y) {
    float max_diff = 0.0f;
    for(int64_t i = 0; i < x->numel; i++) {
        float d = fabsf(x->data[i] - y->data[i]);
        if(d > max_diff) max_diff = d;
    }
    return max_diff;
}

int main(void) {
    pico_init();

    int sizes[] = {128, 256, 512, 1024};
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("\n  pico matmul thread scaling   (warmup=%d, samples=%d median, -O2)\n", WARMUP,
           SAMPLES);
    printf("  %-8s %12s %12s %12s %10s %10s\n", "N", "scalar ms", "avx ms", "speedup",
           "avx GF/s", "diff");
    printf("  ----------------------------------------------------------------------\n");

    for(int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        int64_t shape[] = {n, n};
        struct PicoTensor* a = pico_param(shape, 2);
        struct PicoTensor* b = pico_param(shape, 2);
        struct PicoTensor* out = pico_param(shape, 2);
        struct PicoTensor* ref = pico_param(shape, 2);

        for(int64_t i = 0; i < a->numel; i++) {
            a->data[i] = (float)((i % 13) - 6) * 0.25f;
            b->data[i] = (float)((i % 7) - 3) * 0.5f;
        }

        size_t bytes = (size_t)out->numel * sizeof(float);
        memset(ref->data, 0, bytes);
        pico_matmul_cpu_scalar(a, b, ref);
        memset(out->data, 0, bytes);
        pico_matmul_cpu_avx(a, b, out);

        float diff = max_abs_diff(out, ref);
        double scalar_t = median_time(pico_matmul_cpu_scalar, a, b, out);
        double avx_t = median_time(pico_matmul_cpu_avx, a, b, out);
        double flops = 2.0 * (double)n * (double)n * (double)n;
        double avx_g = flops / avx_t / 1e9;

        printf("  %-8d %12.3f %12.3f %11.2fx %10.2f %10.3e%s\n", n, scalar_t * 1e3,
               avx_t * 1e3, scalar_t / avx_t, avx_g, diff, diff <= TOL ? "" : " MISMATCH");

        pico_free(a);
        pico_free(b);
        pico_free(out);
        pico_free(ref);
    }

    printf("\n");
    return 0;
}
