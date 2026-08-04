#include "global.h"

#include <stdio.h>
#include <stdlib.h>  // Required for rand() and srand()
#include <time.h>    // Required for time()

#include "arena.h"
#include "ctx.h"
#include "tpool.h"

SimdLevel g_simd_level = SIMD_NONE;
GpuBackend g_gpu_backend = GPU_UNKNOWN;
int g_pico_initialized = 0;
uint32_t x_state = 123456789;  // Ultra-fast state variables (non-zero seeds)
struct PicoTPool* global_tp = NULL;

// the ONE real definition of the arena ctx stack (declared extern in arena.h)
thread_local struct Arena* arena_stack[MAX_ARENA_STACK];
thread_local int arena_stack_top = -1;

static SimdLevel detect_simd(void) {
    return SIMD_AVX2;
}

static GpuBackend detect_gpu(void) {
    return GPU_CUDA;
}

static void pico_print_banner(void) {
    printf("\n\n");
    printf("  ════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  ████  ███  ███   ███    \n");
    printf("  █░░░█  █░░█ ░░░ █ ░░█   \n");
    printf("  ████░░ █░░█░ ░░░█░ ░█░  \n");
    printf("  █░░░░ ░█░░█░░   █░░ █░░ \n");
    printf("  █░░░░░███░ ███   ███ ░░ \n");
    printf("   ░░    ░░░  ░░░   ░░░ ░ \n");
    printf("    ░     ░░░  ░░░   ░░░  \n");
    printf("\n");
    printf("  ════════════════════════════════════════════════════════════\n");
    printf("\n");
}

// INFO: pico_init_verbose starts one pico run and gives you the run context.
// pass false in tests/benches so init does not spam the banner/logs per case.
struct PicoContext* pico_init_verbose(bool verbose) {
    if(g_pico_initialized) {
        if(verbose) {
            fprintf(stderr, "PicoRuntimeError: pico is already initialized\n");
        }
        return NULL;
    }

    struct PicoContext* ctx = malloc(sizeof(struct PicoContext));
    if(ctx == NULL) {
        fprintf(stderr, "PicoRuntimeError: failed to allocate pico context\n");
        return NULL;
    }

    *ctx = pico_context_init();
    if(ctx->arena == NULL) {
        free(ctx);
        fprintf(stderr, "PicoRuntimeError: failed to initialize pico context\n");
        return NULL;
    }

    srand(time(NULL));  // seed random numbers, thankssssss

    g_simd_level = detect_simd();
    g_gpu_backend = detect_gpu();
    g_pico_initialized = 1;

    if(verbose) {
        pico_print_banner();
    }

    x_state = (uint32_t)time(NULL);

    global_tp = pico_tpool_create(8);
    if(global_tp != NULL) {
        if(verbose) {
            printf("Initialized the global thread pool");
            printf("\n");
        }
    } else {
        fprintf(stderr, "PicoThreadPoolError: failed to initialize global thread pool\n");
    }

    return ctx;
}

// INFO: default init is noisy on purpose, so examples/users see pico started.
struct PicoContext* pico_init(void) {
    return pico_init_verbose(true);
}

// INFO: shutdown closes the whole run: context memory first, then global runtime pieces.
void pico_shutdown(struct PicoContext* ctx) {
    if(ctx != NULL) {
        pico_context_destroy(ctx);
        free(ctx);
    }

    pico_tpool_destroy(global_tp);
    global_tp = NULL;
    g_pico_initialized = 0;
}
