#pragma once

#include "kernels/matmul/avx2_16x_exec.h"
#include "kernels/matmul/avx2_8x8_exec.h"
#include "tensor.h"

__attribute__((target("avx2,fma"), always_inline)) static inline void pico_matmul_cpu_avx_exec(
    struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out, int row_start, int row_end, int columns,
    int k_dim) {
    pico_matmul_cpu_avx_16x_exec(a, b, out, row_start, row_end, columns, k_dim);
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx(struct PicoTensor* a, struct PicoTensor* b,
                                                                           struct PicoTensor* out) {
    pico_matmul_cpu_avx_16x(a, b, out);
}
