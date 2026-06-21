#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"

struct PicoTensor* pico_param(int64_t* shape, uint8_t ndim) {
    struct PicoTensor* tensor = (struct PicoTensor*)calloc(1, sizeof(struct PicoTensor));
    if(tensor == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->is_persistent = 1;

    // allocate and copy the shape array
    tensor->shape = (int64_t*)calloc(ndim, sizeof(int64_t));
    if(tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));

    // compute number of elements
    int numel = pico_compute_numel(tensor->shape, tensor->ndim);

    tensor->data = (float*)calloc(numel, sizeof(float));
    tensor->grad = (float*)calloc(numel, sizeof(float));
    tensor->strides = (int64_t*)calloc(tensor->ndim, sizeof(int64_t));

    // check if any inner allocations failed
    if(tensor->data == NULL || tensor->grad == NULL || tensor->strides == NULL) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor->grad);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    tensor->numel = numel;
    // compute strides using the freshly allocated array
    pico_compute_strides(shape, ndim, tensor->strides);

    return tensor;
}

struct PicoTensor* pico_create_tensor(struct Arena* arena, int64_t* shape, uint8_t ndim) {
    struct PicoTensor* tensor = (struct PicoTensor*)arena_alloc(arena, sizeof(struct PicoTensor));
    if(tensor == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->is_persistent = 0;

    // arena_alloc returns GARBAGE (not zeroed like calloc), so init these by hand
    // or the op/autograd code will read junk pointers.
    tensor->_backward = NULL;
    tensor->parents = NULL;
    tensor->num_parents = 0;
    tensor->backend = CPU;  // ops override this to inherit from inputs

    // allocate and copy the shape array
    tensor->shape = (int64_t*)arena_alloc(arena, (ndim * sizeof(int64_t)));
    if(tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));

    // compute number of elements
    int numel = pico_compute_numel(tensor->shape, tensor->ndim);

    tensor->data = (float*)arena_alloc(arena, numel * sizeof(float));
    tensor->grad = (float*)arena_alloc(arena, numel * sizeof(float));
    tensor->strides = (int64_t*)arena_alloc(arena, tensor->ndim * sizeof(int64_t));

    // check if any inner allocations failed
    if(tensor->data == NULL || tensor->grad == NULL || tensor->strides == NULL) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor->grad);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    tensor->numel = numel;
    pico_compute_strides(shape, ndim, tensor->strides);

    return tensor;
}

void pico_free(struct PicoTensor* tensor) {
    // if the pointer is already NULL, do nothing safely
    if(tensor == NULL) {
        return;
    }

    // check if memory is in an arena
    if(tensor->is_persistent == 0) {
        return;
    }

    // free internal arrays first
    if(tensor->shape != NULL) {
        free(tensor->shape);
    }
    if(tensor->strides != NULL) {
        free(tensor->strides);
    }
    if(tensor->data != NULL) {
        free(tensor->data);
    }
    if(tensor->grad != NULL) {
        free(tensor->grad);
    }

    // free the dynamic parent array if it was allocated
    if(tensor->parents != NULL) {
        free(tensor->parents);
    }

    // free the main tensor structure
    free(tensor);
}

uint8_t pico_check_broadcast_compatibility(struct PicoTensor* a, struct PicoTensor* b) {
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;

    // We check from the end of the shape arrays (the "trailing" dimensions)
    int i = ndim_a - 1;
    int j = ndim_b - 1;

    while(i >= 0 && j >= 0) {
        int dim_a = a->shape[i];
        int dim_b = b->shape[j];

        // The Broadcasting Rule:
        // 1. Dimensions are equal, OR
        // 2. One of them is 1
        if(dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return 0;  // Not compatible!
        }
        i--;
        j--;
    }

    // If one tensor has more dimensions (e.g., [5, 4, 3] vs [4, 3]),
    // the extra leading dimensions [5] are always compatible with
    // the "implicit ones" of the smaller tensor.
    return 1;
}
