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
struct Arena* global_arena = NULL;
static int g_pico_shutdown_registered = 0;
static int g_global_arena_pushed = 0;

// the ONE real definition of the arena ctx stack (declared extern in arena.h)
thread_local struct Arena* arena_stack[MAX_ARENA_STACK];
thread_local int arena_stack_top = -1;

static SimdLevel detect_simd(void) {
    return SIMD_AVX2;
}

static GpuBackend detect_gpu(void) {
    return GPU_CUDA;
}

// INFO: pico_init owns the default process-level services. right now that means
// simd/backend detection, the global tpool, and a default arena so tiny examples
// can pass NULL without writing setup code.
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

    global_arena = arena_init(PICO_DEFAULT_ARENA_SIZE);
    if(global_arena != NULL) {
        // NOTE: only push the default arena if the current thread has no arena.
        // if the caller already pushed one, NULL should keep meaning their arena,
        // not silently switch to pico's global arena.
        if(arena_ctx_current() == NULL) {
            arena_ctx_push(global_arena);
            g_global_arena_pushed = 1;
        }
        printf("Initialized the global arena allocator (%d MiB)", PICO_DEFAULT_ARENA_SIZE / (1024 * 1024));
        printf("\n");
    } else {
        fprintf(stderr, "PicoArenaError: failed to initialize global arena\n");
    }
}

// INFO: shutdown mirrors init. the default arena is popped only if pico_init was
// the one that pushed it, which keeps user-managed ctx stacks balanced.
void pico_shutdown(void) {
    if(global_arena != NULL) {
        if(g_global_arena_pushed && arena_ctx_current() == global_arena) {
            arena_ctx_pop();
        }
        arena_destroy(global_arena);
        global_arena = NULL;
        g_global_arena_pushed = 0;
    }

    pico_tpool_destroy(global_tp);
    global_tp = NULL;
    g_pico_initialized = 0;
}
