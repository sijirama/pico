#pragma once

#include <stdint.h>
typedef enum { SIMD_NONE, SIMD_SSE, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;
typedef enum { GPU_UNKNOWN, GPU_OPENCL } GpuBackend;

extern SimdLevel g_simd_level;
extern GpuBackend g_gpu_backend;
extern int g_pico_initialized;




extern uint32_t x_state;
#define PI_F 3.14159265358979323846f  // M_PI isn't exposed under -std=c11



void pico_init(void);

