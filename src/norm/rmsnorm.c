#include "norm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../arena.h"
#include "../ctx.h"
#include "../devices/backend.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"

static void pico_rmsnorm_backward(struct PicoTensor* self) {
    struct PicoTensor* input = self->parents[0];
    struct PicoTensor* weight = self->parents[1];
    struct PicoTensor* eps_t = self->parents[2];

    int D = input->shape[input->ndim - 1];
    int64_t rows = input->numel / D;
    float eps = eps_t->data[0];

    for(int64_t row = 0; row < rows; row++) {
        int64_t base = row * D;
        float mean_sq = 0.0f;
        for(int d = 0; d < D; d++) {
            float x = input->data[base + d];
            mean_sq += x * x;
        }

        mean_sq /= (float)D;
        float rms = sqrtf(mean_sq + eps);
        float inv_rms = 1.0f / rms;
        float inv_rms_cubed = inv_rms * inv_rms * inv_rms;

        float dot = 0.0f;
        for(int d = 0; d < D; d++) {
            dot += self->grad[base + d] * weight->data[d] * input->data[base + d];
        }

        for(int d = 0; d < D; d++) {
            float upstream_times_weight = self->grad[base + d] * weight->data[d];
            input->grad[base + d] += upstream_times_weight * inv_rms -
                                     input->data[base + d] * dot * inv_rms_cubed / (float)D;
            weight->grad[d] += self->grad[base + d] * input->data[base + d] * inv_rms;
        }
    }
}

struct PicoRMSNorm* pico_nn_rmsnorm_init(struct PicoContext* ctx, char* name, int normalized_dim, float eps) {
    if(ctx == NULL || name == NULL || normalized_dim <= 0) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for rmsnorm init allocation\n");
        return NULL;
    }

    struct PicoRMSNorm* norm = malloc(sizeof(struct PicoRMSNorm));
    if(norm == NULL) {
        perror("Failed to allocate PicoRMSNorm");
        return NULL;
    }

    norm->name = arena_alloc(arena, strlen(name) + 1);
    if(norm->name == NULL) {
        free(norm);
        return NULL;
    }
    strcpy(norm->name, name);

    size_t weight_name_len = strlen(name) + strlen(".weight") + 1;
    char* weight_name = arena_alloc(arena, weight_name_len);
    if(weight_name == NULL) {
        free(norm);
        return NULL;
    }
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");

    int64_t weight_shape[] = {normalized_dim};
    norm->weight = pico_param_named(ctx, weight_name, weight_shape, 1);
    if(norm->weight == NULL) {
        free(norm);
        return NULL;
    }

    for(int i = 0; i < normalized_dim; i++) {
        norm->weight->data[i] = 1.0f;
    }

    norm->normalized_dim = normalized_dim;
    norm->eps = eps > 0.0f ? eps : 1e-5f;

    return norm;
}

struct PicoTensor* pico_nn_rmsnorm_forward(struct PicoContext* ctx, struct PicoRMSNorm* norm,
                                           struct PicoTensor* input) {
    if(ctx == NULL || norm == NULL || input == NULL) {
        return NULL;
    }

    if(input->ndim < 1 || input->shape[input->ndim - 1] != norm->normalized_dim) {
        fprintf(stderr, "PicoNormError: rmsnorm input last dim must match normalized_dim\n");
        return NULL;
    }

    if(!pico_require_same_backend(input, norm->weight, "rmsnorm")) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for rmsnorm forward allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor_on(ctx, input->backend, input->shape, input->ndim);
    struct PicoTensor* eps_t = pico_tensor_from_scalar_on(ctx, input->backend, norm->eps);
    if(out == NULL || eps_t == NULL) {
        return NULL;
    }

    if(input->backend == PICO_BACKEND_CPU) {
        int D = norm->normalized_dim;
        int64_t rows = input->numel / D;

        for(int64_t row = 0; row < rows; row++) {
            int64_t base = row * D;
            float mean_sq = 0.0f;
            for(int d = 0; d < D; d++) {
                float x = input->data[base + d];
                mean_sq += x * x;
            }

            mean_sq /= (float)D;
            float inv_rms = 1.0f / sqrtf(mean_sq + norm->eps);

            for(int d = 0; d < D; d++) {
                out->data[base + d] = input->data[base + d] * inv_rms * norm->weight->data[d];
            }
        }
    } else if(input->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_rmsnorm(input, norm->weight, out, norm->eps)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: rmsnorm unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 3);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = input;
    out->parents[1] = norm->weight;
    out->parents[2] = eps_t;
    out->num_parents = 3;
    out->_backward = pico_rmsnorm_backward;

    return out;
}

void pico_nn_rmsnorm_free(struct PicoRMSNorm* norm) {
    if(norm == NULL) {
        return;
    }

    free(norm);
}
