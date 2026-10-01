#include "tensor_ops.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"
#include "autograd.h"
#include "ctx.h"
#include "kernels/cuda/cuda_ops.h"
#include "tensor.h"

void pico_transpose_2d(struct PicoTensor *tensor) {
    if(tensor->ndim != 2) {
        fprintf(stderr, "Error: This is not a rank 2 tensor!\n");
        return;
    }

    if(!pico_require_cpu_backend(tensor->backend, "transpose")) {
        return;
    }

    // swap rows/cols in metadata only. the underlying data buffer is unchanged.
    int c = tensor->shape[1];
    tensor->shape[1] = tensor->shape[0];
    tensor->shape[0] = c;

    int sc = tensor->strides[1];
    tensor->strides[1] = tensor->strides[0];
    tensor->strides[0] = sc;
}

struct PicoTensor *
pico_cat(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b, int dim) {
    if(!pico_require_same_backend(a, b, "cat")) {
        return NULL;
    }
    if(!pico_require_cpu_backend(a->backend, "cat")) {
        return NULL;
    }
    if(a->ndim != b->ndim) {
        fprintf(
            stderr,
            "[Pico] Error: PicoTensors are not compatible for contatenation, Mismatch found in "
            "ndim!\n");
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for cat allocation\n");
        return NULL;
    }

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * a->ndim);

    // dim=0 means stack them over each other dim=1 means side by side
    for(int i = 0; i < a->ndim; i++) {
        if(i == dim) {
            res_shape[i] = a->shape[i] + b->shape[i];
            continue;
        }
        if(a->shape[i] != b->shape[i]) {
            fprintf(
                stderr,
                "[Pico] Error: PicoTensors are not compatible for contatenation, Mismatch "
                "found in shape!\n");
            return NULL;
        }
        res_shape[i] = a->shape[i];
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, a->ndim);

    float *src_a = (float *)a->data;
    float *src_b = (float *)b->data;
    float *dst = (float *)out->data;

    int64_t outer_count = 1;
    for(int i = 0; i < dim; i++) {
        outer_count *= a->shape[i];
    }

    int64_t inner_size = 1;
    for(int i = dim + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }

    int64_t a_copy_size = a->shape[dim] * inner_size;
    int64_t b_copy_size = b->shape[dim] * inner_size;

    for(int64_t o = 0; o < outer_count; o++) {
        // 1. Copy chunk from tensor A
        memcpy(dst, src_a, a_copy_size * sizeof(float));
        dst += a_copy_size;
        src_a += a_copy_size;

        // 2. Copy chunk from tensor B right next to it
        memcpy(dst, src_b, b_copy_size * sizeof(float));
        dst += b_copy_size;
        src_b += b_copy_size;
    }

    return out;
}

struct PicoTensor *pico_clone(struct PicoContext *ctx, struct PicoTensor *tensor) {
    if(!pico_require_cpu_backend(tensor->backend, "clone")) {
        return NULL;
    }

    struct PicoTensor *t = pico_tensor_from_data_on(ctx, tensor->backend, tensor->shape, tensor->ndim, tensor->data);
    return t;
}

void pico_view(struct PicoContext *ctx, struct PicoTensor *tensor, int64_t *shape, int ndim) {
    if(tensor->kind != PICO_TENSOR_TEMP) {
        fprintf(stderr, "view only supports temp tensors for now\n");
        return;
    }

    if(!pico_require_cpu_backend(tensor->backend, "view")) {
        return;
    }

    int new_numel = pico_compute_numel(shape, ndim);
    if(new_numel != tensor->numel) {
        fprintf(stderr, "view shape must have same numel\n");
        return;
    }

    struct Arena *arena = pico_context_temp_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no temp arena available for view allocation\n");
        return;
    }

    int64_t *newShape = (int64_t *)arena_alloc(arena, (ndim * sizeof(int64_t)));
    if(newShape == NULL) {
        return;
    }

    int64_t *newStrides = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    if(newStrides == NULL) {
        return;
    }

    memcpy(newShape, shape, ndim * sizeof(int64_t));
    pico_compute_strides(newShape, ndim, newStrides);
    tensor->shape = newShape;
    tensor->strides = newStrides;
    tensor->ndim = ndim;
}

// NOTE: written by codex
// TODO: come back to this later siji

void pico_permute(struct PicoContext *ctx, struct PicoTensor *tensor, int64_t *axes) {
    if(ctx == NULL || tensor == NULL || axes == NULL) {
        return;
    }

    if(tensor->kind != PICO_TENSOR_TEMP) {
        fprintf(stderr, "permute only supports temp tensors for now\n");
        return;
    }

    if(!pico_require_cpu_backend(tensor->backend, "permute")) {
        return;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for permute allocation\n");
        return;
    }

    bool seen[256] = {false};
    for(int d = 0; d < tensor->ndim; d++) {
        if(axes[d] < 0 || axes[d] >= tensor->ndim || seen[axes[d]]) {
            fprintf(stderr, "permute axes must be a valid dimension permutation\n");
            return;
        }
        seen[axes[d]] = true;
    }

    int ndim = tensor->ndim;
    int64_t *newShape = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    int64_t *newStrides = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    int64_t *oldLogicalStrides = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    int64_t *oldCoords = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    int64_t *newCoords = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    float *newData = (float *)arena_alloc(arena, tensor->numel * sizeof(float));
    if(newShape == NULL || newStrides == NULL || oldLogicalStrides == NULL || oldCoords == NULL ||
       newCoords == NULL || newData == NULL) {
        return;
    }

    for(int d = 0; d < ndim; d++) {
        newShape[d] = tensor->shape[axes[d]];
    }
    pico_compute_strides(newShape, ndim, newStrides);
    pico_compute_strides(tensor->shape, ndim, oldLogicalStrides);

    for(int64_t logical_i = 0; logical_i < tensor->numel; logical_i++) {
        int64_t rem = logical_i;
        for(int d = 0; d < ndim; d++) {
            oldCoords[d] = rem / oldLogicalStrides[d];
            rem %= oldLogicalStrides[d];
        }

        for(int d = 0; d < ndim; d++) {
            newCoords[d] = oldCoords[axes[d]];
        }

        int64_t oldOffset = 0;
        int64_t newOffset = 0;
        for(int d = 0; d < ndim; d++) {
            oldOffset += oldCoords[d] * tensor->strides[d];
            newOffset += newCoords[d] * newStrides[d];
        }

        newData[newOffset] = tensor->data[oldOffset];
    }

    tensor->shape = newShape;
    tensor->strides = newStrides;
    tensor->data = newData;
}

struct PicoTensor *pico_dropout(struct PicoContext *ctx, struct PicoTensor *tensor, float p) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(p < 0.0f || p >= 1.0f) {
        fprintf(stderr, "dropout p must be in [0, 1)\n");
        return NULL;
    }

    if(!pico_require_cpu_backend(tensor->backend, "dropout")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for dropout allocation\n");
        return NULL;
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, tensor->shape, tensor->ndim);
    struct PicoTensor *mask = pico_create_tensor_on(ctx, tensor->backend, tensor->shape, tensor->ndim);
    if(out == NULL || mask == NULL) {
        return NULL;
    }

    if(ctx->mode == PICO_EVAL || p == 0.0f) {
        for(int64_t i = 0; i < tensor->numel; i++) {
            mask->data[i] = 1.0f;
            out->data[i] = tensor->data[i];
        }
    } else {
        struct PicoTensor *random = pico_rand(ctx, tensor->shape, tensor->ndim);
        if(random == NULL) {
            return NULL;
        }

        float scale = 1.0f / (1.0f - p);
        for(int64_t i = 0; i < tensor->numel; i++) {
            mask->data[i] = random->data[i] >= p ? scale : 0.0f;
            out->data[i] = tensor->data[i] * mask->data[i];
        }
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = tensor;
    out->parents[1] = mask;
    out->num_parents = 2;
    out->_backward = pico_dropout_backward;

    return out;
}

struct PicoTensor *pico_softmax(struct PicoContext *ctx, struct PicoTensor *tensor, uint8_t dim) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim >= tensor->ndim) {
        fprintf(stderr, "softmax dim is out of range\n");
        return NULL;
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, tensor->shape, tensor->ndim);
    if(out == NULL) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for softmax graph allocation\n");
        return NULL;
    }

    if(tensor->backend == PICO_BACKEND_CPU) {
        int ndim = tensor->ndim;
        int64_t axis_len = tensor->shape[dim];
        int64_t slice_count = tensor->numel / axis_len;

        for(int64_t slice_i = 0; slice_i < slice_count; slice_i++) {
            int64_t rem = slice_i;
            int64_t input_base = 0;
            int64_t output_base = 0;

            for(int d = ndim - 1; d >= 0; d--) {
                if(d == dim) {
                    continue;
                }

                int64_t coord = rem % tensor->shape[d];
                rem /= tensor->shape[d];
                input_base += coord * tensor->strides[d];
                output_base += coord * out->strides[d];
            }

            float max_value = tensor->data[input_base];
            for(int64_t i = 1; i < axis_len; i++) {
                float value = tensor->data[input_base + i * tensor->strides[dim]];
                if(value > max_value) {
                    max_value = value;
                }
            }

            float sum = 0.0f;
            for(int64_t i = 0; i < axis_len; i++) {
                float value = expf(tensor->data[input_base + i * tensor->strides[dim]] - max_value);
                out->data[output_base + i * out->strides[dim]] = value;
                sum += value;
            }

            for(int64_t i = 0; i < axis_len; i++) {
                out->data[output_base + i * out->strides[dim]] /= sum;
            }
        }
    } else if(tensor->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_softmax(tensor, out, dim)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: softmax unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *));
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = tensor;
    out->num_parents = 1;
    out->op_param = dim;
    out->_backward = pico_softmax_backward;

    return out;
}

static bool pico_apply_causal_mask_cpu(struct PicoTensor *tensor, int window) {
    if(tensor == NULL || tensor->data == NULL) {
        return false;
    }

    if(tensor->ndim != 3 && tensor->ndim != 4) {
        fprintf(stderr, "PicoAttentionError: causal softmax expects a 3D or 4D tensor\n");
        return false;
    }

    int64_t B = tensor->ndim == 4 ? tensor->shape[0] : 1;
    int64_t H = tensor->ndim == 4 ? tensor->shape[1] : tensor->shape[0];
    int64_t Q = tensor->ndim == 4 ? tensor->shape[2] : tensor->shape[1];
    int64_t K = tensor->ndim == 4 ? tensor->shape[3] : tensor->shape[2];

    if(Q != K) {
        fprintf(stderr, "PicoAttentionError: causal softmax expects square attention scores\n");
        return false;
    }

    for(int64_t b = 0; b < B; b++) {
        for(int64_t h = 0; h < H; h++) {
            for(int64_t q = 0; q < Q; q++) {
                for(int64_t k = 0; k < K; k++) {
                    int64_t offset = tensor->ndim == 4
                                         ? b * tensor->strides[0] + h * tensor->strides[1] + q * tensor->strides[2] + k * tensor->strides[3]
                                         : h * tensor->strides[0] + q * tensor->strides[1] + k * tensor->strides[2];

                    bool outside_window = window >= 0 && k < q - window;
                    if(k > q || outside_window) {
                        tensor->data[offset] = -INFINITY;
                    }
                }
            }
        }
    }

    return true;
}

struct PicoTensor *pico_causal_softmax(struct PicoContext *ctx, struct PicoTensor *tensor, uint8_t dim, int window) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim >= tensor->ndim) {
        fprintf(stderr, "causal softmax dim is out of range\n");
        return NULL;
    }

    if(tensor->backend == PICO_BACKEND_CPU) {
        if(!pico_apply_causal_mask_cpu(tensor, window)) {
            return NULL;
        }
        return pico_softmax(ctx, tensor, dim);
    }

    if(tensor->backend != PICO_BACKEND_CUDA) {
        fprintf(stderr, "PicoBackendError: causal softmax unknown backend\n");
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for causal softmax graph allocation\n");
        return NULL;
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, tensor->shape, tensor->ndim);
    if(out == NULL) {
        return NULL;
    }

    if(!pico_cuda_causal_softmax(tensor, out, dim, window)) {
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 1);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = tensor;
    out->num_parents = 1;
    out->op_param = dim;
    out->_backward = pico_softmax_backward;

    return out;
}

struct PicoTensor *pico_sum(struct PicoContext *ctx, struct PicoTensor *tensor, int dim) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim < -1 || dim >= tensor->ndim) {
        fprintf(stderr, "sum dim is out of range\n");
        return NULL;
    }

    if(!pico_require_cpu_backend(tensor->backend, "sum")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sum allocation\n");
        return NULL;
    }

    if(dim == -1) {
        int64_t scalar_shape[1] = {1};
        struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, scalar_shape, 1);
        if(out == NULL) {
            return NULL;
        }

        float sum = 0.0f;
        for(int64_t i = 0; i < tensor->numel; i++) {
            sum += tensor->data[i];
        }
        out->data[0] = sum;
        return out;
    }

    uint8_t out_ndim = tensor->ndim == 1 ? 1 : tensor->ndim - 1;
    int64_t *out_shape = arena_alloc(arena, sizeof(int64_t) * out_ndim);
    if(out_shape == NULL) {
        return NULL;
    }

    if(tensor->ndim == 1) {
        out_shape[0] = 1;
    } else {
        int out_d = 0;
        for(int d = 0; d < tensor->ndim; d++) {
            if(d != dim) {
                out_shape[out_d++] = tensor->shape[d];
            }
        }
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, out_shape, out_ndim);
    if(out == NULL) {
        return NULL;
    }

    int64_t axis_len = tensor->shape[dim];
    for(int64_t out_i = 0; out_i < out->numel; out_i++) {
        int64_t rem = out_i;
        int64_t input_base = 0;

        for(int d = tensor->ndim - 1; d >= 0; d--) {
            if(d == dim) {
                continue;
            }

            int64_t out_shape_d = tensor->shape[d];
            int64_t coord = rem % out_shape_d;
            rem /= out_shape_d;
            input_base += coord * tensor->strides[d];
        }

        float sum = 0.0f;
        for(int64_t i = 0; i < axis_len; i++) {
            sum += tensor->data[input_base + i * tensor->strides[dim]];
        }
        out->data[out_i] = sum;
    }

    return out;
}

struct PicoTensor *pico_mean(struct PicoContext *ctx, struct PicoTensor *tensor, int dim) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim < -1 || dim >= tensor->ndim) {
        fprintf(stderr, "mean dim is out of range\n");
        return NULL;
    }

    if(tensor->backend == PICO_BACKEND_CPU) {
        struct PicoTensor *out = pico_sum(ctx, tensor, dim);
        if(out == NULL) {
            return NULL;
        }

        float divisor = dim == -1 ? (float)tensor->numel : (float)tensor->shape[dim];
        for(int64_t i = 0; i < out->numel; i++) {
            out->data[i] /= divisor;
        }

        return out;
    } else if(tensor->backend == PICO_BACKEND_CUDA) {
        struct Arena *arena = pico_context_arena(ctx);
        if(arena == NULL) {
            fprintf(stderr, "PicoArenaError: no arena available for mean allocation\n");
            return NULL;
        }

        if(dim == -1) {
            int64_t scalar_shape[1] = {1};
            struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, scalar_shape, 1);
            if(out == NULL || !pico_cuda_mean(tensor, out, dim)) {
                return NULL;
            }
            return out;
        }

        uint8_t out_ndim = tensor->ndim == 1 ? 1 : tensor->ndim - 1;
        int64_t *out_shape = arena_alloc(arena, sizeof(int64_t) * out_ndim);
        if(out_shape == NULL) {
            return NULL;
        }

        if(tensor->ndim == 1) {
            out_shape[0] = 1;
        } else {
            int out_d = 0;
            for(int d = 0; d < tensor->ndim; d++) {
                if(d != dim) {
                    out_shape[out_d++] = tensor->shape[d];
                }
            }
        }

        struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, out_shape, out_ndim);
        if(out == NULL || !pico_cuda_mean(tensor, out, dim)) {
            return NULL;
        }
        return out;
    }

    fprintf(stderr, "PicoBackendError: mean unknown backend\n");
    return NULL;
}

struct PicoTensor *pico_var(struct PicoContext *ctx, struct PicoTensor *tensor, int dim) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim < -1 || dim >= tensor->ndim) {
        fprintf(stderr, "var dim is out of range\n");
        return NULL;
    }

    if(!pico_require_cpu_backend(tensor->backend, "var")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for var allocation\n");
        return NULL;
    }

    if(dim == -1) {
        int64_t scalar_shape[1] = {1};
        struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, scalar_shape, 1);
        if(out == NULL) {
            return NULL;
        }

        float mean = 0.0f;
        for(int64_t i = 0; i < tensor->numel; i++) {
            mean += tensor->data[i];
        }
        mean /= (float)tensor->numel;

        float var = 0.0f;
        for(int64_t i = 0; i < tensor->numel; i++) {
            float diff = tensor->data[i] - mean;
            var += diff * diff;
        }
        out->data[0] = var / (float)tensor->numel;
        return out;
    }

    uint8_t out_ndim = tensor->ndim == 1 ? 1 : tensor->ndim - 1;
    int64_t *out_shape = arena_alloc(arena, sizeof(int64_t) * out_ndim);
    if(out_shape == NULL) {
        return NULL;
    }

    if(tensor->ndim == 1) {
        out_shape[0] = 1;
    } else {
        int out_d = 0;
        for(int d = 0; d < tensor->ndim; d++) {
            if(d != dim) {
                out_shape[out_d++] = tensor->shape[d];
            }
        }
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, tensor->backend, out_shape, out_ndim);
    if(out == NULL) {
        return NULL;
    }

    int64_t axis_len = tensor->shape[dim];
    for(int64_t out_i = 0; out_i < out->numel; out_i++) {
        int64_t rem = out_i;
        int64_t input_base = 0;

        for(int d = tensor->ndim - 1; d >= 0; d--) {
            if(d == dim) {
                continue;
            }

            int64_t out_shape_d = tensor->shape[d];
            int64_t coord = rem % out_shape_d;
            rem /= out_shape_d;
            input_base += coord * tensor->strides[d];
        }

        float mean = 0.0f;
        for(int64_t i = 0; i < axis_len; i++) {
            mean += tensor->data[input_base + i * tensor->strides[dim]];
        }
        mean /= (float)axis_len;

        float var = 0.0f;
        for(int64_t i = 0; i < axis_len; i++) {
            float diff = tensor->data[input_base + i * tensor->strides[dim]] - mean;
            var += diff * diff;
        }
        out->data[out_i] = var / (float)axis_len;
    }

    return out;
}
