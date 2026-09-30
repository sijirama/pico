#include "ops.h"

#include <stdint.h>
#include <stdio.h>

#include "arena.h"
#include "autograd.h"
#include "ctx.h"
#include "devices/backend.h"
#include "global.h"
#include "kernels/cpu/cpu_kernels.h"
#include "kernels/cuda/cuda_ops.h"
#include "tensor.h"

struct PicoTensor *pico_add(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "add")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for add allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t *a_padded_shape = pad_shape(ctx, a, ndim);
    int64_t *b_padded_shape = pad_shape(ctx, b, ndim);

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, ndim);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_add_cpu(a, b, out);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_add(a, b, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: add unknown backend\n");
        return NULL;
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_add_backward;

    return out;
}

struct PicoTensor *pico_sub(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "sub")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sub allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t *a_padded_shape = pad_shape(ctx, a, ndim);
    int64_t *b_padded_shape = pad_shape(ctx, b, ndim);

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, ndim);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_sub_cpu(a, b, out);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_sub(a, b, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: sub unknown backend\n");
        return NULL;
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_sub_backward;

    return out;
}

struct PicoTensor *pico_mul(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "mul")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for mul allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t *a_padded_shape = pad_shape(ctx, a, ndim);
    int64_t *b_padded_shape = pad_shape(ctx, b, ndim);

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, ndim);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_mul_cpu(a, b, out);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_mul(a, b, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: mul unknown backend\n");
        return NULL;
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_mul_backward;

    return out;
}

struct PicoTensor *pico_div(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "div")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for div allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t *a_padded_shape = pad_shape(ctx, a, ndim);
    int64_t *b_padded_shape = pad_shape(ctx, b, ndim);

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, ndim);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_div_cpu(a, b, out);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_div(a, b, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: div unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_div_backward;

    return out;
}

struct PicoTensor *pico_matmul(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b) {

    bool is_2d_matmul = a->ndim == 2 && b->ndim == 2;
    bool is_3d_2d_matmul = a->ndim == 3 && b->ndim == 2;
    bool is_3d_matmul = a->ndim == 3 && b->ndim == 3;
    bool is_4d_matmul = a->ndim == 4 && b->ndim == 4;

    if(!is_2d_matmul && !is_3d_2d_matmul && !is_3d_matmul && !is_4d_matmul) {
        fprintf(stderr, "[Pico] Error: matmul only supports 2D, 3D@2D, 3D, or 4D tensors\n");
        return NULL;
    }

    if(is_2d_matmul && a->shape[1] != b->shape[0]) {
        perror("[Pico] Error: 2 matmuls matrices must be compatible");
        return NULL;
    }

    if(is_3d_2d_matmul && a->shape[2] != b->shape[0]) {
        perror("[Pico] Error: 3d@2d batched matmul tensors must be compatible");
        return NULL;
    }

    if(is_3d_matmul && (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1])) {
        perror("[Pico] Error: 3d batched matmul tensors must be compatible");
        return NULL;
    }

    if(is_4d_matmul && (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1] || a->shape[3] != b->shape[2])) {
        perror("[Pico] Error: 4d batched matmul tensors must be compatible");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "matmul")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for matmul allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int rows = is_4d_matmul ? a->shape[2] : (a->ndim == 3 ? a->shape[1] : a->shape[0]);
    int columns = is_4d_matmul ? b->shape[3] : (b->ndim == 3 ? b->shape[2] : b->shape[1]);

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    if(is_2d_matmul) {
        res_shape[0] = rows;
        res_shape[1] = columns;
    } else if(is_3d_2d_matmul) {
        res_shape[0] = a->shape[0];
        res_shape[1] = a->shape[1];
        res_shape[2] = columns;
    } else if(is_3d_matmul) {
        res_shape[0] = a->shape[0];
        res_shape[1] = rows;
        res_shape[2] = columns;
    } else {
        res_shape[0] = a->shape[0];
        res_shape[1] = a->shape[1];
        res_shape[2] = rows;
        res_shape[3] = columns;
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, ndim);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_matmul_cpu(a, b, out);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_matmul(a, b, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: matmul unknown backend\n");
        return NULL;
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_matmul_backward;

    return out;
}

struct PicoTensor *pico_swa_matmul(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b, int windows) {
    return pico_matmul(ctx, a, b);
}

struct PicoTensor *pico_grouped_matmul(struct PicoContext *ctx, struct PicoTensor *a, struct PicoTensor *b,
                                       int group_size) {
    if(a == NULL || b == NULL) {
        fprintf(stderr, "[Pico] Error: grouped_matmul received NULL tensor\n");
        return NULL;
    }

    if(group_size <= 0) {
        fprintf(stderr, "[Pico] Error: grouped_matmul group_size must be positive\n");
        return NULL;
    }

    if(a->ndim != 4 || b->ndim != 4) {
        fprintf(stderr, "[Pico] Error: grouped_matmul only supports 4D tensors\n");
        return NULL;
    }

    if(a->shape[0] != b->shape[0] || a->shape[3] != b->shape[2]) {
        fprintf(stderr, "[Pico] Error: grouped_matmul tensors must have compatible batch and inner dims\n");
        return NULL;
    }

    if(a->shape[1] != b->shape[1] * group_size) {
        fprintf(stderr, "[Pico] Error: grouped_matmul requires Hq == Hkv * group_size\n");
        return NULL;
    }

    if(!pico_require_same_backend(a, b, "grouped_matmul")) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for grouped_matmul allocation\n");
        return NULL;
    }

    int64_t *res_shape = arena_alloc(arena, sizeof(int64_t) * 4);
    if(res_shape == NULL) {
        return NULL;
    }

    res_shape[0] = a->shape[0];
    res_shape[1] = a->shape[1];
    res_shape[2] = a->shape[2];
    res_shape[3] = b->shape[3];

    struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, res_shape, 4);
    if(out == NULL) {
        return NULL;
    }

    if(a->backend == PICO_BACKEND_CPU) {
        pico_grouped_matmul_cpu(a, b, out, group_size);
    } else if(a->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_grouped_matmul(a, b, out, group_size)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: grouped_matmul unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->op_param = group_size;
    out->_backward = pico_grouped_matmul_backward;

    return out;
}

// ---- unary element-wise math ----------------------------------------------
// same shape as `out`, dispatch to the CPU kernel, wire the single parent so
// the graph stays intact. unary => num_parents == 1. these are near-identical:
// prime for a later bundle.

#define PICO_DEFINE_UNARY_OP(name)                                                       \
    struct PicoTensor *pico_##name(struct PicoContext *ctx, struct PicoTensor *a) {       \
        struct Arena *arena = pico_context_arena(ctx);                                    \
        if(arena == NULL) {                                                              \
            fprintf(stderr, "PicoArenaError: no arena available for " #name " allocation\n"); \
            return NULL;                                                                 \
        }                                                                                \
        struct PicoTensor *out = pico_create_tensor_on(ctx, a->backend, a->shape, a->ndim); \
        if(out == NULL) {                                                                \
            return NULL;                                                                 \
        }                                                                                \
        if(a->backend == PICO_BACKEND_CPU) {                                             \
            pico_##name##_cpu(a, out);                                                   \
        } else if(a->backend == PICO_BACKEND_CUDA) {                                     \
            if(!pico_cuda_##name(a, out)) {                                              \
                return NULL;                                                             \
            }                                                                            \
        } else {                                                                         \
            fprintf(stderr, "PicoBackendError: " #name " unknown backend\n");           \
            return NULL;                                                                 \
        }                                                                                \
        out->parents = arena_alloc(arena, sizeof(struct PicoTensor *));                  \
        out->parents[0] = a;                                                             \
        out->num_parents = 1;                                                            \
        out->_backward = pico_##name##_backward;                                         \
        return out;                                                                      \
    }

PICO_DEFINE_UNARY_OP(sqrt)
PICO_DEFINE_UNARY_OP(sin)
PICO_DEFINE_UNARY_OP(cos)
PICO_DEFINE_UNARY_OP(tan)
PICO_DEFINE_UNARY_OP(tanh)
PICO_DEFINE_UNARY_OP(log)
