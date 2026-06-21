#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    tensor->grad = (float*)calloc(1, sizeof(float));
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

    // compute strides using the freshly allocated array
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
