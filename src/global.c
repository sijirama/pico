#include "global.h"

#include <stdio.h>
#include <stdlib.h>  // Required for rand() and srand()
#include <time.h>    // Required for time()

#include "arena.h"
#include "tpool.h"

SimdLevel g_simd_level = SIMD_NONE;
GpuBackend g_gpu_backend = GPU_UNKNOWN;
int g_pico_initialized = 0;
uint32_t x_state = 123456789;  // Ultra-fast state variables (non-zero seeds)
struct PicoTPool* global_tp = NULL;
static int g_pico_shutdown_registered = 0;

// the ONE real definition of the arena ctx stack (declared extern in arena.h)
thread_local struct Arena* arena_stack[MAX_ARENA_STACK];
thread_local int arena_stack_top = -1;

static SimdLevel detect_simd(void) {
    return SIMD_AVX2;
}

static GpuBackend detect_gpu(void) {
    return GPU_CUDA;
}

void pico_init(void) {
    if(g_pico_initialized)
        return;

    srand(time(NULL));  // seed random numbers, thankssssss

    g_simd_level = detect_simd();
    g_gpu_backend = detect_gpu();
    g_pico_initialized = 1;
    if(!g_pico_shutdown_registered) {
        if(atexit(pico_shutdown) != 0) {
            fprintf(stderr, "PicoThreadPoolError: failed to register pico_shutdown at exit\n");
        } else {
            g_pico_shutdown_registered = 1;
        }
    }

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

    x_state = (uint32_t)time(NULL);

    global_tp = pico_tpool_create(8);
    if(global_tp != NULL) {
        printf("Initialized the global thread pool");
        printf("\n");
    } else {
        fprintf(stderr, "PicoThreadPoolError: failed to initialize global thread pool\n");
    }
}

void pico_shutdown(void) {
    pico_tpool_destroy(global_tp);
    global_tp = NULL;
    g_pico_initialized = 0;
}
