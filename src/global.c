#include "global.h"

#include <stdio.h>
#include <stdlib.h>  // Required for rand() and srand()
#include <time.h>    // Required for time()

#include "arena.h"

SimdLevel g_simd_level = SIMD_NONE;
GpuBackend g_gpu_backend = GPU_UNKNOWN;
int g_pico_initialized = 0;

// the ONE real definition of the arena ctx stack (declared extern in arena.h)
thread_local struct Arena* arena_stack[MAX_ARENA_STACK];
thread_local int arena_stack_top = -1;

static SimdLevel detect_simd(void) {
    return SIMD_AVX2;
}

static GpuBackend detect_gpu(void) {
    return GPU_OPENCL;
}

void pico_init(void) {
    if(g_pico_initialized)
        return;

    srand(time(NULL));  // seed random numbers, thankssssss

    g_simd_level = detect_simd();
    g_gpu_backend = detect_gpu();
    g_pico_initialized = 1;

    printf("[PICO] Pico ready.\n");
}
