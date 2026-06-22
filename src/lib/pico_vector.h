#pragma once

#include <stdint.h>

#include "tensor.h"

struct PicoTensorDVector {
    struct PicoTensor** data;  // Pointer to the array of picotensor pointers
    size_t size;               // Current number of elements stored
    size_t capacity;           // Total capacity allocated
};

static inline void init_pico_tensor_d_vector(struct PicoTensorDVector* a, size_t initialCapacity) {
    a->data = malloc(initialCapacity * sizeof(struct PicoTensor*));
    if(a->data == NULL) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    a->size = 0;
    a->capacity = initialCapacity;
}

static inline void insert_pico_tensor_d_vector(struct PicoTensorDVector* a,
                                               struct PicoTensor* element) {
    if(a->size == a->capacity) {
        // Double the capacity when full
        size_t newCapacity = a->capacity * 2;

        // Use a temporary pointer to avoid memory loss if realloc fails
        struct PicoTensor** temp = realloc(a->data, newCapacity * sizeof(struct PicoTensor*));
        if(temp == NULL) {
            perror("Reallocation failed");
            // Original memory is still valid, handle gracefully or exit
            free(a->data);
            exit(EXIT_FAILURE);
        }
        a->data = temp;
        a->capacity = newCapacity;
    }
    // Store the element and increment size
    a->data[a->size++] = element;
}

static inline int search_pico_tensor_d_vector(struct PicoTensorDVector* a,
                                              struct PicoTensor* element) {
    if(a == NULL) {
        return -2;
    }
    for (int i = 0; i < a->size; i++) {
        if(a->data[i] == element){
            return i;
        }
    }
    return -1;
}

// reverse in place: two pointers from the ends, swap and walk inward.
static inline void reverse_pico_tensor_d_vector(struct PicoTensorDVector* a) {
    if(a == NULL || a->size < 2) {
        return;  // nothing to reverse (also guards size-1 underflow below)
    }
    size_t i = 0;
    size_t j = a->size - 1;
    while(i < j) {
        struct PicoTensor* tmp = a->data[i];
        a->data[i] = a->data[j];
        a->data[j] = tmp;
        i++;
        j--;
    }
}

static inline void free_pico_tensor_d_vector(struct PicoTensorDVector* a) {
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}
