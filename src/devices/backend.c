#include "backend.h"

#include <stdio.h>
#include <string.h>

#include "../arena.h"
#include "../ctx.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"

const char* pico_backend_name(PicoBackend backend) {
    switch(backend) {
        case PICO_BACKEND_CPU:
            return "CPU";
        case PICO_BACKEND_CUDA:
            return "CUDA";
        default:
            return "UNKNOWN";
    }
}

bool pico_require_same_backend(struct PicoTensor* a, struct PicoTensor* b, const char* op_name) {
    if(a->backend != b->backend) {
        fprintf(
            stderr,
            "PicoBackendError: %s received tensors on different backends: %s and %s\n",
            op_name,
            pico_backend_name(a->backend),
            pico_backend_name(b->backend));
        return false;
    }
    return true;
}

bool pico_require_cpu_backend(PicoBackend backend, const char* op_name) {
    if(backend != PICO_BACKEND_CPU) {
        fprintf(stderr, "PicoBackendError: %s %s backend is not implemented yet\n", op_name, pico_backend_name(backend));
        return false;
    }
    return true;
}

static struct Arena* pico_tensor_data_arena(struct PicoContext* ctx, struct PicoTensor* tensor) {
    if(tensor->kind == PICO_TENSOR_PARAM) {
        return pico_context_param_arena(ctx);
    }
    return pico_context_temp_arena(ctx);
}

bool pico_tensor_init_data_on(struct PicoContext* ctx, struct PicoTensor* tensor, PicoBackend backend) {
    if(ctx == NULL || tensor == NULL) {
        return false;
    }

    if(backend == PICO_BACKEND_CUDA) {
        return pico_cuda_tensor_alloc(ctx, tensor);
    }

    if(backend != PICO_BACKEND_CPU) {
        fprintf(stderr, "PicoBackendError: unknown backend %s\n", pico_backend_name(backend));
        return false;
    }

    struct Arena* arena = pico_tensor_data_arena(ctx, tensor);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for %s tensor data allocation\n",
                tensor->kind == PICO_TENSOR_PARAM ? "param" : "temp");
        return false;
    }

    tensor->data = (float*)arena_alloc(arena, tensor->numel * sizeof(float));
    tensor->grad = (float*)arena_alloc(arena, tensor->numel * sizeof(float));
    if(tensor->data == NULL || tensor->grad == NULL) {
        return false;
    }

    memset(tensor->data, 0, tensor->numel * sizeof(float));
    memset(tensor->grad, 0, tensor->numel * sizeof(float));
    tensor->backend = PICO_BACKEND_CPU;
    return true;
}

bool pico_tensor_to_backend(struct PicoContext* ctx, struct PicoTensor* tensor, PicoBackend backend) {
    if(ctx == NULL || tensor == NULL) {
        return false;
    }

    if(tensor->backend == backend) {
        return true;
    }

    if(backend == PICO_BACKEND_CUDA) {
        return pico_cuda_tensor_to_cuda(ctx, tensor);
    }

    if(backend != PICO_BACKEND_CPU) {
        fprintf(stderr, "PicoBackendError: unknown backend %s\n", pico_backend_name(backend));
        return false;
    }

    float* old_data = tensor->data;
    float* old_grad = tensor->grad;
    PicoBackend old_backend = tensor->backend;

    if(!pico_tensor_init_data_on(ctx, tensor, PICO_BACKEND_CPU)) {
        tensor->backend = old_backend;
        tensor->data = old_data;
        tensor->grad = old_grad;
        return false;
    }

    if(old_backend == PICO_BACKEND_CUDA) {
        return pico_cuda_tensor_to_cpu(ctx, tensor, old_data, old_grad);
    }

    if(old_data != NULL) {
        memcpy(tensor->data, old_data, tensor->numel * sizeof(float));
    }
    if(old_grad != NULL) {
        memcpy(tensor->grad, old_grad, tensor->numel * sizeof(float));
    }

    return true;
}
