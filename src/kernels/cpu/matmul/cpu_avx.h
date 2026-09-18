#pragma once

#include "../../../tensor.h"
#include "avx2_16x_exec.h"
#include "avx2_8x8_exec.h"

__attribute__((target("avx2,fma"), always_inline)) static inline void
pico_matmul_cpu_avx_exec(
    struct PicoTensor *a,
    struct PicoTensor *b,
    struct PicoTensor *out,
    int row_start,
    int row_end,
    int columns,
    int k_dim) {
    pico_matmul_cpu_avx_16x_exec(a, b, out, row_start, row_end, columns, k_dim);
}

__attribute__((target("avx2,fma"))) static inline void pico_matmul_cpu_avx(
    struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {

    // 2D path
    if(a->ndim == 2) {
        pico_matmul_cpu_avx_16x(a, b, out);
        return;
    }

    // Batched projection path:
    // A   [B, M, K]
    // B   [K, N]
    // OUT [B, M, N]
    if(a->ndim == 3 && b->ndim == 2) {
        int64_t a_shape[2] = {a->shape[1], a->shape[2]};
        int64_t out_shape[2] = {out->shape[1], out->shape[2]};

        int64_t a_strides[2] = {a->strides[1], a->strides[2]};
        int64_t out_strides[2] = {out->strides[1], out->strides[2]};

        for(int64_t batch = 0; batch < a->shape[0]; batch++) {
            struct PicoTensor a_view = *a;
            struct PicoTensor out_view = *out;

            a_view.ndim = 2;
            out_view.ndim = 2;

            a_view.shape = a_shape;
            out_view.shape = out_shape;

            a_view.strides = a_strides;
            out_view.strides = out_strides;

            a_view.data = a->data + batch * a->strides[0];
            out_view.data = out->data + batch * out->strides[0];

            pico_matmul_cpu_avx_16x(&a_view, b, &out_view);
        }
        return;
    }

    // Batched 3D path:
    // A   [B, M, K]
    // B   [B, K, N]
    // OUT [B, M, N]
    if(a->ndim == 3 && b->ndim == 3) {
        int64_t a_shape[2] = {a->shape[1], a->shape[2]};
        int64_t b_shape[2] = {b->shape[1], b->shape[2]};
        int64_t out_shape[2] = {out->shape[1], out->shape[2]};

        int64_t a_strides[2] = {a->strides[1], a->strides[2]};
        int64_t b_strides[2] = {b->strides[1], b->strides[2]};
        int64_t out_strides[2] = {out->strides[1], out->strides[2]};

        for(int64_t batch = 0; batch < a->shape[0]; batch++) {
            struct PicoTensor a_view = *a;
            struct PicoTensor b_view = *b;
            struct PicoTensor out_view = *out;

            a_view.ndim = 2;
            b_view.ndim = 2;
            out_view.ndim = 2;

            a_view.shape = a_shape;
            b_view.shape = b_shape;
            out_view.shape = out_shape;

            a_view.strides = a_strides;
            b_view.strides = b_strides;
            out_view.strides = out_strides;

            a_view.data = a->data + batch * a->strides[0];
            b_view.data = b->data + batch * b->strides[0];
            out_view.data = out->data + batch * out->strides[0];

            pico_matmul_cpu_avx_16x(&a_view, &b_view, &out_view);
        }
        return;
    }

    // Batched attention path:
    // A   [B, H, M, K]
    // B   [B, H, K, N]
    // OUT [B, H, M, N]
    if(a->ndim == 4) {

        int64_t a_shape[2] = {a->shape[2], a->shape[3]};

        int64_t b_shape[2] = {b->shape[2], b->shape[3]};

        int64_t out_shape[2] = {out->shape[2], out->shape[3]};

        int64_t a_strides[2] = {a->strides[2], a->strides[3]};

        int64_t b_strides[2] = {b->strides[2], b->strides[3]};

        int64_t out_strides[2] = {out->strides[2], out->strides[3]};

        for(int64_t batch = 0; batch < a->shape[0]; batch++) {
            for(int64_t head = 0; head < a->shape[1]; head++) {

                struct PicoTensor a_view = *a;
                struct PicoTensor b_view = *b;
                struct PicoTensor out_view = *out;

                a_view.ndim = 2;
                b_view.ndim = 2;
                out_view.ndim = 2;

                a_view.shape = a_shape;
                b_view.shape = b_shape;
                out_view.shape = out_shape;

                a_view.strides = a_strides;
                b_view.strides = b_strides;
                out_view.strides = out_strides;

                a_view.data =
                    a->data + batch * a->strides[0] + head * a->strides[1];

                b_view.data =
                    b->data + batch * b->strides[0] + head * b->strides[1];

                out_view.data = out->data + batch * out->strides[0] +
                                head * out->strides[1];

                pico_matmul_cpu_avx_16x(&a_view, &b_view, &out_view);
            }
        }
        return;
    }
}
