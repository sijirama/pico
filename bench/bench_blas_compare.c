/*
 * bench_blas_compare: compare Pico AVX matmul against a CBLAS SGEMM provider.
 *
 * Build through bench/Makefile targets:
 *   make blas_openblas
 *   make blas_blis
 */
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

#ifndef BLAS_NAME
#define BLAS_NAME "CBLAS"
#endif

#define WARMUP 1
#define SAMPLES 3
#define TOL 1e-3f

typedef void (*pico_matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

struct shape {
    const char* name;
    int m;
    int k;
    int n;
};

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

static void fill_tensor(struct PicoTensor* t, int mod, float scale) {
    for(int64_t i = 0; i < t->numel; i++) {
        t->data[i] = (float)((i % mod) - (mod / 2)) * scale;
    }
}

static float max_abs_diff(struct PicoTensor* x, struct PicoTensor* y) {
    float max_diff = 0.0f;
    for(int64_t i = 0; i < x->numel; i++) {
        float d = fabsf(x->data[i] - y->data[i]);
        if(d > max_diff)
            max_diff = d;
    }
    return max_diff;
}

static void blas_sgemm(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a->data, k, b->data, n,
                0.0f, out->data, n);
}

static double median_time_pico(pico_matmul_fn fn, struct PicoTensor* a, struct PicoTensor* b,
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

static double median_time_blas(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    double samples[SAMPLES];

    for(int w = 0; w < WARMUP; w++) {
        blas_sgemm(a, b, out);
    }

    for(int s = 0; s < SAMPLES; s++) {
        double t0 = now_sec();
        blas_sgemm(a, b, out);
        samples[s] = now_sec() - t0;
    }

    qsort(samples, SAMPLES, sizeof(double), cmp_double);
    return samples[SAMPLES / 2];
}

int main(void) {
    pico_init();

    struct shape shapes[] = {
        {"512^3", 512, 512, 512},
        {"768^3", 768, 768, 768},
        {"1024^3", 1024, 1024, 1024},
        {"1536^3", 1536, 1536, 1536},
        {"512x1024x2048", 512, 1024, 2048},
        {"2048x1024x512", 2048, 1024, 512},
    };
    int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n  pico vs %s SGEMM   (warmup=%d, samples=%d median, -O2)\n", BLAS_NAME, WARMUP,
           SAMPLES);
    printf("  %-18s %10s %10s %10s %10s %10s\n", "shape", "pico ms", "pico G/s",
           "blas ms", "blas G/s", "diff");
    printf("  ------------------------------------------------------------------------\n");

    for(int s = 0; s < n_shapes; s++) {
        struct shape shape = shapes[s];
        int64_t sa[] = {shape.m, shape.k};
        int64_t sb[] = {shape.k, shape.n};
        int64_t so[] = {shape.m, shape.n};

        struct PicoTensor* a = pico_param(sa, 2);
        struct PicoTensor* b = pico_param(sb, 2);
        struct PicoTensor* pico_out = pico_param(so, 2);
        struct PicoTensor* blas_out = pico_param(so, 2);

        fill_tensor(a, 13, 0.25f);
        fill_tensor(b, 7, 0.5f);

        memset(pico_out->data, 0, (size_t)pico_out->numel * sizeof(float));
        pico_matmul_cpu_avx(a, b, pico_out);
        blas_sgemm(a, b, blas_out);
        float diff = max_abs_diff(pico_out, blas_out);

        double pico_t = median_time_pico(pico_matmul_cpu_avx, a, b, pico_out);
        double blas_t = median_time_blas(a, b, blas_out);
        double flops = 2.0 * (double)shape.m * (double)shape.k * (double)shape.n;

        printf("  %-18s %10.3f %10.2f %10.3f %10.2f %10.3e%s\n", shape.name,
               pico_t * 1e3, flops / pico_t / 1e9, blas_t * 1e3, flops / blas_t / 1e9, diff,
               diff > TOL ? " MISMATCH" : "");

        pico_free(a);
        pico_free(b);
        pico_free(pico_out);
        pico_free(blas_out);
    }

    printf("\n");
    return 0;
}
