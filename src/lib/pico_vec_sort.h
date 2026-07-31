#pragma once

#include "lib/pico_vector.h"

// INFO: char-specific vec sort. this assumes each element points to a char, or
// to a string where the first byte is the char we care about. it sorts the
// pointer slots in place, not the chars themselves.
static inline void pico_vec_sort_chars(struct PicoVec* vec) {
    if(vec == NULL || vec->size < 2) {
        return;
    }

    for(size_t i = 1; i < vec->size; i++) {
        void* current = vec->data[i];
        unsigned char current_char = *(unsigned char*)current;
        size_t j = i;

        while(j > 0 && *(unsigned char*)vec->data[j - 1] > current_char) {
            vec->data[j] = vec->data[j - 1];
            j--;
        }

        vec->data[j] = current;
    }
}
