/*
 * bench_matmul_large: larger AVX matmul shapes for checking threaded/tpool scaling.
 *
 * Run with `make matmul_large` from bench/.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

#define WARMUP 1
#define SAMPLES 3
#define TOL 1e-3f

typedef void (*matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

struct shape {
    const char* name;
    int m;
    int k;
    int n;
    int check_scalar;
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
        if(d > max_diff)
            max_diff = d;
    }
    return max_diff;
}

static void fill_tensor(struct PicoTensor* t, int mod, float scale) {
    for(int64_t i = 0; i < t->numel; i++) {
        t->data[i] = (float)((i % mod) - (mod / 2)) * scale;
    }
}

int main(void) {
    pico_init();

    struct shape shapes[] = {
        {"square 768^3", 768, 768, 768, 1},
        {"square 1024^3", 1024, 1024, 1024, 0},
        {"square 1536^3", 1536, 1536, 1536, 0},
        {"wide 512x1024x2048", 512, 1024, 2048, 0},
        {"tall 2048x1024x512", 2048, 1024, 512, 0},
    };
    int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n  pico large matmul benchmark   (warmup=%d, samples=%d median, -O2)\n", WARMUP,
           SAMPLES);
    printf("  %-22s %12s %12s %12s\n", "shape", "avx ms", "avx GF/s", "diff");
    printf("  --------------------------------------------------------------\n");

    for(int s = 0; s < n_shapes; s++) {
        struct shape shape = shapes[s];
        int64_t sa[] = {shape.m, shape.k};
        int64_t sb[] = {shape.k, shape.n};
        int64_t so[] = {shape.m, shape.n};

        struct PicoTensor* a = pico_param(sa, 2);
        struct PicoTensor* b = pico_param(sb, 2);
        struct PicoTensor* out = pico_param(so, 2);
        struct PicoTensor* ref = NULL;

        fill_tensor(a, 13, 0.25f);
        fill_tensor(b, 7, 0.5f);

        float diff = 0.0f;
        if(shape.check_scalar) {
            ref = pico_param(so, 2);
            memset(ref->data, 0, (size_t)ref->numel * sizeof(float));
            pico_matmul_cpu_scalar(a, b, ref);
            memset(out->data, 0, (size_t)out->numel * sizeof(float));
            pico_matmul_cpu_avx(a, b, out);
            diff = max_abs_diff(out, ref);
        }

        double avx_t = median_time(pico_matmul_cpu_avx, a, b, out);
        double flops = 2.0 * (double)shape.m * (double)shape.k * (double)shape.n;
        double avx_g = flops / avx_t / 1e9;

        printf("  %-22s %12.3f %12.2f %12.3e%s\n", shape.name, avx_t * 1e3, avx_g,
               diff, shape.check_scalar && diff > TOL ? " MISMATCH" : "");

        pico_free(a);
        pico_free(b);
        pico_free(out);
        if(ref != NULL)
            pico_free(ref);
    }

    printf("\n");
    return 0;
}
