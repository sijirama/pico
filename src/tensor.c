#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "lib/pico_vector.h"

void postorder(struct PicoTensor* root, struct PicoVec* vector,
               struct PicoVec* visited);

void pico_backward(struct Arena* arena, struct PicoTensor* entry) {
    // build our dependency graph with dfs
    struct PicoVec vector, visited;
    pico_vec_init(&vector, 25);
    pico_vec_init(&visited, 25);
    postorder(entry, &vector, &visited);

    // post-order gives [leaves ... entry]; reverse -> [entry ... leaves]
    pico_vec_reverse(&vector);

    // seed the entry node with grad 1
    struct PicoTensor* curr = NULL;
    curr = (struct PicoTensor*)vector.data[0];
    for(int i = 0; i < curr->numel; i++) {
        curr->grad[i] = 1.0f;
    }

    // call backward on each  (now iterate FORWARD: entry is first)
    for(int i = 0; i < vector.size; i++) {
        curr = (struct PicoTensor*)vector.data[i];
        if(curr->_backward != NULL) {
            curr->_backward(curr);
        }
    }

    pico_vec_free(&vector);
    pico_vec_free(&visited);
}

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
    memset(tensor->data, 0, numel * sizeof(float));
    tensor->grad = (float*)arena_alloc(arena, numel * sizeof(float));
    memset(tensor->grad, 0, numel * sizeof(float));
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

void postorder(struct PicoTensor* root, struct PicoVec* vector,
               struct PicoVec* visited) {
    if(root == NULL) {
        return;
    }

    for(int i = 0; i < root->num_parents; i++) {
        postorder(root->parents[i], vector, visited);
    }

    // append to array if not appended before
    if(pico_vec_find(visited, root) == -1) {
        pico_vec_push(vector, root);
        pico_vec_push(visited, root);
    }
}
