#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "tpool.h"

struct PicoContext;

typedef enum { SIMD_NONE, SIMD_SSE, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;
typedef enum { GPU_UNKNOWN, GPU_OPENCL, GPU_CUDA } GpuBackend;

extern SimdLevel g_simd_level;
extern GpuBackend g_gpu_backend;
extern int g_pico_initialized;

extern struct PicoTPool* global_tp;

extern uint32_t x_state;
#define PI_F 3.14159265358979323846f  // M_PI isn't exposed under -std=c11

struct PicoContext* pico_init(void);
struct PicoContext* pico_init_verbose(bool verbose);
void pico_shutdown(struct PicoContext* ctx);
