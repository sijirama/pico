#pragma once

typedef enum { SIMD_NONE, SIMD_SSE, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;
typedef enum { GPU_UNKNOWN, GPU_OPENCL } GpuBackend;

extern SimdLevel g_simd_level;
extern GpuBackend g_gpu_backend;
extern int g_pico_initialized;

void pico_init(void);
