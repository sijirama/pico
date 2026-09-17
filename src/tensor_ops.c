#include "tensor_ops.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "tensor.h"

void pico_transpose_2d(struct PicoTensor *tensor) {
    if(tensor->ndim != 2) {
        fprintf(stderr, "Error: This is not a rank 2 tensor!\n");
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
    if(a->backend != b->backend) {
        fprintf(
            stderr,
            "[Pico] Error: PicoTensor backends are not compatible, Mismatch found in backends!\n");
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

    struct PicoTensor *out = pico_create_tensor(ctx, res_shape, a->ndim);

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
    struct PicoTensor *t = pico_tensor_from_data(ctx, tensor->shape, tensor->ndim, tensor->data);
    return t;
}

void pico_view(struct PicoContext *ctx, struct PicoTensor *tensor, int64_t *shape, int ndim) {
    if(tensor->storage != PICO_TENSOR_STORAGE_ARENA) {
        fprintf(stderr, "view only supports arena tensors for now\n");
        return;
    }

    int new_numel = pico_compute_numel(shape, ndim);
    if(new_numel != tensor->numel) {
        fprintf(stderr, "view shape must have same numel\n");
        return;
    }

    int64_t *newShape = (int64_t *)arena_alloc(ctx->arena, (ndim * sizeof(int64_t)));
    if(newShape == NULL) {
        return;
    }

    int64_t *newStrides = (int64_t *)arena_alloc(ctx->arena, ndim * sizeof(int64_t));
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

    if(tensor->storage != PICO_TENSOR_STORAGE_ARENA) {
        fprintf(stderr, "permute only supports arena tensors for now\n");
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

struct PicoTensor *pico_softmax(struct PicoContext *ctx, struct PicoTensor *tensor, uint8_t dim) {
    if(ctx == NULL || tensor == NULL) {
        return NULL;
    }

    if(dim >= tensor->ndim) {
        fprintf(stderr, "softmax dim is out of range\n");
        return NULL;
    }

    struct PicoTensor *out = pico_create_tensor(ctx, tensor->shape, tensor->ndim);
    if(out == NULL) {
        return NULL;
    }

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

    return out;
}
