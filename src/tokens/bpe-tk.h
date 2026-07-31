#pragma once

#include <ctype.h>
#include <stdint.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "lib/pico_map.h"
#include "lib/pico_vector.h"
#include "tokens/tokenizer.h"

struct BPEPicoTKData {
    struct PicoHashMap* vocab;  // char * | size_t
};

static const struct TokenizerVTable BPE_TK_METHODS = {};

// ===================== these are util functions for the bpe to operate

static inline void convert_to_lowercase(char* str) {
    // Loop until the null-terminator '\0' is reached
    while(*str != '\0') {
        *str = tolower((unsigned char)*str);  // Convert current char
        str++;                                // Move pointer to next char
    }
}

static inline void bpe_pretokenize(struct PicoVec* container, char* normalized_text) {
    // Delimiters for tokenization: whitespace and punctuation
    const char* delimiters = " \t\n\r.,;:!?()[]{}<>\"'`~@#$%^&*-+=|/";

    char* text_copy = malloc(strlen(normalized_text) + 1);
    if(text_copy == NULL) {
        perror("malloc failed");
        return;
    }
    strcpy(text_copy, normalized_text);

    char* token = strtok(text_copy, delimiters);
    while(token != NULL) {
        // Allocate memory for the token string
        char* token_copy = malloc(strlen(token) + 1);
        if(token_copy == NULL) {
            perror("malloc failed");
            free(text_copy);
            return;
        }
        strcpy(token_copy, token);

        // Append the token to the container
        pico_vec_push(container, token_copy);

        token = strtok(NULL, delimiters);
    }

    free(text_copy);
}

// INFO: we want this to text to sort of be like the entry way to adding voacb into the tokenizer
// this has to be a bpe tokenizer specificaclly
static inline void bpe_ingest_text(struct Tokenizer* tokenizer, char* text_input) {
    if(tokenizer == NULL || tokenizer->data == NULL || text_input == NULL) {
        return;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    if(data->vocab == NULL) {
        return;
    }

    struct PicoVec temp_container;
    pico_vec_init(&temp_container, 16);

    convert_to_lowercase(text_input);
    bpe_pretokenize(&temp_container, text_input);

    // our temp_container should have values we want to append to the hash
    // loop through the values in temp_ontainer to use eachc of them
    char* current_string;
    for(size_t i = 0; i < temp_container.size; i++) {
        current_string = (char*)temp_container.data[i];
        //
        // if not in vocab, add token with freq 1
        if(pico_hashmap_contains(data->vocab, current_string) == false) {
            pico_hashmap_insert(data->vocab, current_string, (void*)(uintptr_t)1);
        } else {
            // if in vocab, inccrement the value (the frequency)
            struct PicoHashEntry* entry = pico_hashmap_find_entry(data->vocab, current_string);
            size_t new_freq = (size_t)(uintptr_t)entry->value + 1;
            entry->value = (void*)(uintptr_t)new_freq;
        }

        free(current_string);
    }

    // kill the temp_container
    pico_vec_free(&temp_container);
}

// INFO: unlike the wordbased tokenizer, this tk is gonna initiate a tokenizer then have another function to fill in the
// tokens, so this function will be to return a tokenizer, another function will be used to fill up the vocab and ids
static inline struct Tokenizer* pico_bpe_tk_init(struct PicoContext* context) {
    if(context == NULL || context->arena == NULL) {
        return NULL;
    }

    struct Tokenizer* tokenizer = arena_alloc(context->arena, sizeof(struct Tokenizer));
    struct BPEPicoTKData* data = arena_alloc(context->arena, sizeof(struct BPEPicoTKData));
    if(tokenizer == NULL || data == NULL) {
        return NULL;
    }

    // init the data used in the fucking stuff i guess lol
    data->vocab = pico_hashmap_init();

    tokenizer->ctx = context;
    tokenizer->methods = &BPE_TK_METHODS;
    tokenizer->data = data;

    return tokenizer;
}
