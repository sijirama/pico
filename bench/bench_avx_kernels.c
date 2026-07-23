/*
 * bench_avx_kernels — compare matmul microkernel roll widths across matrix shapes.
 *
 * Strategies: scalar, 1x8, 2x8, 4x8, 8x8 (per-roll drivers from bench_common.h),
 * and the real adaptive pico_matmul_cpu_avx (8->4->2->1 cascade). Run with
 * `make avx_kernels` from inside bench/.
 *
 * The point: bigger tile != automatically faster. Once the accumulators +
 * broadcasts exceed the 16 architectural YMM registers the compiler spills, so a
 * smaller roll can win — and the winner changes with matrix shape. Each strategy
 * is correctness-gated against scalar before its time is reported.
 */
#include <stdlib.h>

#include "bench_common.h"

#define WARMUP 3
#define ITERS 10
#define TOL 1e-1f  // FP summation order differs across strategies; flag only real bugs

struct strat {
    const char* name;
    bench_matmul_fn fn;
};

struct shape {
    const char* name;
    int M, K, N;  // (M,K) @ (K,N) -> (M,N)
};

int main(void) {
    pico_init();

    struct strat strats[] = {
        {"scalar", pico_matmul_cpu_scalar}, {"1x8", bench_matmul_roll1},
        {"2x8", bench_matmul_roll2},        {"4x8", bench_matmul_roll4},
        {"8x8", bench_matmul_roll8},        {"adaptive", pico_matmul_cpu_avx},
    };
    int n_strats = (int)(sizeof(strats) / sizeof(strats[0]));

    struct shape shapes[] = {
        {"small square      64x64x64", 64, 64, 64},
        {"div-by-8 square    256^3", 256, 256, 256},
        {"large square       512^3", 512, 512, 512},
        {"tall-skinny   1024x128x64", 1024, 128, 64},
        {"short-wide    64x128x1024", 64, 128, 1024},
        {"has tails       70x70x70", 70, 70, 70},
    };
    int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n  pico AVX matmul microkernel sweep   (warmup=%d, iters=%d, -O2)\n", WARMUP, ITERS);
    printf("  correctness gated against scalar (tol=%.0e). GFLOP/s = 2*M*K*N / time.\n", TOL);

    for(int s = 0; s < n_shapes; s++) {
        int M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;

        int64_t sa[] = {M, K};
        int64_t sb[] = {K, N};
        int64_t so[] = {M, N};
        struct PicoTensor* a = pico_param(sa, 2);
        struct PicoTensor* b = pico_param(sb, 2);
        struct PicoTensor* out = pico_param(so, 2);
        struct PicoTensor* ref = pico_param(so, 2);

        for(int64_t i = 0; i < a->numel; i++) a->data[i] = (float)((i % 13) - 6) * 0.25f;
        for(int64_t i = 0; i < b->numel; i++) b->data[i] = (float)((i % 7) - 3) * 0.5f;

        // reference = scalar full matmul
        memset(ref->data, 0, (size_t)ref->numel * sizeof(float));
        pico_matmul_cpu_scalar(a, b, ref);

        double flops = 2.0 * (double)M * (double)K * (double)N;

        printf("\n  %s\n", shapes[s].name);
        printf("  %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
        printf("  %-10s %12s %12s   %s\n", "strategy", "ms/matmul", "GFLOP/s", "correct");
        printf("  ------------------------------------------------------\n");

        double best_g = 0.0;
        const char* best = "";
        for(int st = 0; st < n_strats; st++) {
            memset(out->data, 0, (size_t)out->numel * sizeof(float));
            strats[st].fn(a, b, out);
            float d = bench_max_abs_diff(out, ref);
            int ok = (d <= TOL);

            double t = bench_time_matmul(strats[st].fn, a, b, out, WARMUP, ITERS);
            double g = flops / t / 1e9;
            if(ok && g > best_g) {
                best_g = g;
                best = strats[st].name;
            }
            printf("  %-10s %12.3f %12.2f   %s\n", strats[st].name, t * 1e3, g,
                   ok ? "ok" : "MISMATCH");
        }
        printf("  ------------------------------------------------------\n");
        printf("  winner: %s (%.2f GFLOP/s)\n", best, best_g);

        pico_free(a);
        pico_free(b);
        pico_free(out);
        pico_free(ref);
    }
    printf("\n");
    return 0;
}
