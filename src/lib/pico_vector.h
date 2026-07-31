#pragma once

#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct PicoVec {
    void** data;      // pointer array, callers own the actual objects
    size_t size;      // Current number of elements stored
    size_t capacity;  // Total capacity allocated
};

static inline void pico_vec_init(struct PicoVec* a, size_t initialCapacity) {
    a->data = malloc(initialCapacity * sizeof(void*));
    if(a->data == NULL) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }
    a->size = 0;
    a->capacity = initialCapacity;
}

static inline void pico_vec_push(struct PicoVec* a, void* element) {
    if(a->size == a->capacity) {
        // Double the capacity when full
        size_t newCapacity = a->capacity * 2;

        // Use a temporary pointer to avoid memory loss if realloc fails
        void** temp = realloc(a->data, newCapacity * sizeof(void*));
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

// INFO: shallow copy. the new vec owns a new pointer array, but the pointed-to
// values are the same objects because PicoVec never owns element memory.
static inline struct PicoVec pico_vec_copy(struct PicoVec* src) {
    struct PicoVec copy = {0};
    if(src == NULL) {
        return copy;
    }

    size_t capacity = src->capacity > 0 ? src->capacity : 1;
    pico_vec_init(&copy, capacity);
    copy.size = src->size;

    if(src->size > 0) {
        memcpy(copy.data, src->data, sizeof(void*) * src->size);
    }

    return copy;
}

// INFO: shallow append. result contains the pointer values from a followed by
// b, but it does not clone or own the actual elements.
static inline struct PicoVec* pico_vec_append(struct PicoVec* a, struct PicoVec* b) {
    struct PicoVec* out = malloc(sizeof(struct PicoVec));
    if(out == NULL) {
        perror("Allocation failed");
        exit(EXIT_FAILURE);
    }

    size_t a_size = a == NULL ? 0 : a->size;
    size_t b_size = b == NULL ? 0 : b->size;
    size_t total = a_size + b_size;
    size_t capacity = total > 0 ? total : 1;

    pico_vec_init(out, capacity);
    out->size = total;

    if(a_size > 0) {
        memcpy(out->data, a->data, sizeof(void*) * a_size);
    }

    if(b_size > 0) {
        memcpy(out->data + a_size, b->data, sizeof(void*) * b_size);
    }

    return out;
}

// can we make this faster ? i mean we have size so we can use a good search algo here ??
static inline int pico_vec_find(struct PicoVec* a, void* element) {
    if(a == NULL) {
        return -2;
    }
    for(int i = 0; i < a->size; i++) {
        if(a->data[i] == element) {
            return i;
        }
    }
    return -1;
}

// reverse in place: two pointers from the ends, swap and walk inward.
static inline void pico_vec_reverse(struct PicoVec* a) {
    if(a == NULL || a->size < 2) {
        return;  // nothing to reverse (also guards size-1 underflow below)
    }
    size_t i = 0;
    size_t j = a->size - 1;
    while(i < j) {
        void* tmp = a->data[i];
        a->data[i] = a->data[j];
        a->data[j] = tmp;
        i++;
        j--;
    }
}

static inline void pico_vec_free(struct PicoVec* a) {
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}
