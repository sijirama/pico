#include "global.h"

#include <stdio.h>

SimdLevel g_simd_level = SIMD_NONE;
GpuBackend g_gpu_backend = GPU_UNKNOWN;
int g_pico_initialized = 0;

static SimdLevel detect_simd(void) {
    return SIMD_AVX512;
}

static GpuBackend detect_gpu(void) {
    return GPU_OPENCL;
}

void pico_init(void) {
    if(g_pico_initialized)
        return;

    g_simd_level = detect_simd();
    g_gpu_backend = detect_gpu();
    g_pico_initialized = 1;

    printf("[PICO] Pico ready.\n");
}
