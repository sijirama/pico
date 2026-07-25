/*
 * bench_matmul_mkl: focused matmul benchmark against Intel oneAPI MKL.
 *
 * Build through:
 *   make matmul_mkl
 *
 * This intentionally reuses bench_matmul_focused.c so the Pico strategies,
 * shapes, timing order, and correctness checks stay identical to OpenBLAS.
 */
#define USE_CBLAS
#define USE_MKL
#define BLAS_NAME "Intel MKL"

#include "bench_matmul_focused.c"
