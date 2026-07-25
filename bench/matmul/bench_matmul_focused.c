/*
 * bench_matmul_focused: cleaner timing for the real matmul contenders.
 *
 * Default target:
 *   make matmul_focused
 *
 * OpenBLAS comparison:
 *   make matmul_focused_openblas
 */
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_CBLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

#ifndef BLAS_NAME
#define BLAS_NAME "OpenBLAS"
#endif

#define WARMUP 2
#define SAMPLES 7
#define TOL 1e-1f

typedef void (*matmul_fn)(struct PicoTensor*, struct PicoTensor*, struct PicoTensor*);

struct shape {
    const char* name;
    int m;
    int k;
    int n;
};

struct strat {
    const char* name;
    matmul_fn fn;
};

struct stats {
    double min;
    double median;
    double max;
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

static double gflops(double flops, double seconds) {
    return flops / seconds / 1e9;
}

static struct stats stats_from_samples(double samples[SAMPLES]) {
    qsort(samples, SAMPLES, sizeof(double), cmp_double);
    return (struct stats){.min = samples[0], .median = samples[SAMPLES / 2], .max = samples[SAMPLES - 1]};
}

static void print_runtime_context(void) {
    const char* omp_threads = getenv("OMP_NUM_THREADS");
    const char* omp_bind = getenv("OMP_PROC_BIND");
    const char* omp_places = getenv("OMP_PLACES");

    printf("  env: OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s\n",
           omp_threads != NULL ? omp_threads : "(unset)", omp_bind != NULL ? omp_bind : "(unset)",
           omp_places != NULL ? omp_places : "(unset)");

#ifdef _OPENMP
    printf("  openmp: max_threads=%d num_procs=%d places=%d\n", omp_get_max_threads(),
           omp_get_num_procs(), omp_get_num_places());
#endif

    cpu_set_t set;
    if(sched_getaffinity(0, sizeof(set), &set) == 0) {
        printf("  affinity:");
        for(int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if(CPU_ISSET(cpu, &set))
                printf(" %d", cpu);
        }
        printf("\n");
    }
}

#ifdef USE_CBLAS
static void blas_sgemm(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a->data, k, b->data, n,
                0.0f, out->data, n);
}
#endif

static void time_strategies(struct strat* strats, int n_strats, struct PicoTensor* a,
                            struct PicoTensor* b, struct PicoTensor** outs,
                            struct stats* results) {
    double samples[n_strats][SAMPLES];
    size_t bytes = (size_t)outs[0]->numel * sizeof(float);

    for(int w = 0; w < WARMUP; w++) {
        for(int st = 0; st < n_strats; st++) {
            memset(outs[st]->data, 0, bytes);
            strats[st].fn(a, b, outs[st]);
        }
    }

    for(int s = 0; s < SAMPLES; s++) {
        for(int offset = 0; offset < n_strats; offset++) {
            int st = (s + offset) % n_strats;
            memset(outs[st]->data, 0, bytes);
            double t0 = now_sec();
            strats[st].fn(a, b, outs[st]);
            samples[st][s] = now_sec() - t0;
        }
    }

    for(int st = 0; st < n_strats; st++) {
        results[st] = stats_from_samples(samples[st]);
    }
}

int main(void) {
    pico_init();

    struct strat strats[] = {
        {"8x8-family", pico_matmul_cpu_avx_8x8},
        {"16x-family", pico_matmul_cpu_avx_16x},
        {"pico-avx", pico_matmul_cpu_avx},
#ifdef USE_CBLAS
        {BLAS_NAME, blas_sgemm},
#endif
    };
    int n_strats = (int)(sizeof(strats) / sizeof(strats[0]));

    struct shape shapes[] = {
        {"256^3", 256, 256, 256},
        {"384^3", 384, 384, 384},
        {"512^3", 512, 512, 512},
        {"640^3", 640, 640, 640},
        {"768^3", 768, 768, 768},
        {"960^3", 960, 960, 960},
        {"1024^3", 1024, 1024, 1024},
        {"1280^3", 1280, 1280, 1280},
        {"wide 512x1024x2048", 512, 1024, 2048},
        {"wide 768x1024x1536", 768, 1024, 1536},
        {"wide 1024x512x2048", 1024, 512, 2048},
        {"tall 2048x1024x512", 2048, 1024, 512},
        {"tall 1536x1024x768", 1536, 1024, 768},
        {"tall 2048x512x1024", 2048, 512, 1024},
    };
    int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n  focused matmul benchmark   (warmup=%d, samples=%d, median shown, -O2)\n", WARMUP,
           SAMPLES);
    print_runtime_context();
#ifdef USE_CBLAS
    printf("  BLAS provider: %s\n", BLAS_NAME);
#endif
    printf("  GFLOP/s columns are min / median / max. Correctness diff is max abs vs reference.\n");

    for(int s = 0; s < n_shapes; s++) {
        struct shape shape = shapes[s];
        int64_t sa[] = {shape.m, shape.k};
        int64_t sb[] = {shape.k, shape.n};
        int64_t so[] = {shape.m, shape.n};

        struct PicoTensor* a = pico_param(sa, 2);
        struct PicoTensor* b = pico_param(sb, 2);
        struct PicoTensor* ref = pico_param(so, 2);
        struct PicoTensor* outs[n_strats];
        struct stats results[n_strats];

        fill_tensor(a, 13, 0.25f);
        fill_tensor(b, 7, 0.5f);
        for(int st = 0; st < n_strats; st++) {
            outs[st] = pico_param(so, 2);
        }

#ifdef USE_CBLAS
        blas_sgemm(a, b, ref);
#else
        memset(ref->data, 0, (size_t)ref->numel * sizeof(float));
        pico_matmul_cpu_avx_16x(a, b, ref);
#endif

        double flops = 2.0 * (double)shape.m * (double)shape.k * (double)shape.n;

        printf("\n  %s  (%dx%d * %dx%d)\n", shape.name, shape.m, shape.k, shape.k, shape.n);
        printf("  %-14s %10s %10s %10s %10s\n", "strategy", "min", "median", "max", "diff");
        printf("  -------------------------------------------------------------\n");

        time_strategies(strats, n_strats, a, b, outs, results);

        for(int st = 0; st < n_strats; st++) {
            float diff = max_abs_diff(outs[st], ref);
            const char* suffix = diff > TOL ? " MISMATCH" : "";
#ifdef USE_CBLAS
            if(strats[st].fn == blas_sgemm) {
                diff = 0.0f;
                suffix = " ref";
            }
#endif

            printf("  %-14s %10.2f %10.2f %10.2f %10.3e%s\n", strats[st].name,
                   gflops(flops, results[st].max), gflops(flops, results[st].median),
                   gflops(flops, results[st].min), diff, suffix);
        }

        pico_free(a);
        pico_free(b);
        pico_free(ref);
        for(int st = 0; st < n_strats; st++) {
            pico_free(outs[st]);
        }
    }

    printf("\n");
    return 0;
}
