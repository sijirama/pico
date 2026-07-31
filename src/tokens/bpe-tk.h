#pragma once

#include <stdbool.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "lib/pico_map.h"
#include "lib/pico_vec_sort.h"
#include "lib/pico_vector.h"
#include "tokens/tokenizer.h"

#define MAX_BPE_VOCAB_CAPACITY 1280

struct BPEPicoTKData {
    struct PicoHashMap* corpus;  // char * | size_t
    struct PicoVec* vocab;
    int max_vocab_capacity;
};

static inline size_t pico_bpe_tk_len(const struct Tokenizer* self) {
    if(!self || !self->data)
        return 0;
    const struct BPEPicoTKData* data = (const struct BPEPicoTKData*)self->data;
    return data->corpus->size;
}

static const struct TokenizerVTable BPE_TK_METHODS = {
    .len = pico_bpe_tk_len,
};

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
    if(data->corpus == NULL) {
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
        // if not in corpus, add token with freq 1
        if(pico_hashmap_contains(data->corpus, current_string) == false) {
            pico_hashmap_insert(data->corpus, current_string, (void*)(uintptr_t)1);
        } else {
            // if in corpus, inccrement the value (the frequency)
            struct PicoHashEntry* entry = pico_hashmap_find_entry(data->corpus, current_string);
            size_t new_freq = (size_t)(uintptr_t)entry->value + 1;
            entry->value = (void*)(uintptr_t)new_freq;
        }

        free(current_string);
    }

    // kill the temp_container
    pico_vec_free(&temp_container);
}

// train the vocab and add special tokens too
static inline void bpe_train_vocab(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    // compute the base vocabulary, formed by all the characters used in the corpus:
    struct PicoVec alphabet;
    pico_vec_init(&alphabet, 30);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    if(data->corpus == NULL || data->vocab == NULL) {
        pico_vec_free(&alphabet);
        return;
    }

    struct PicoHashEntry entry;
    bool seen_chars[256] = {0};
    char* p;
    for(size_t i = 0; i < data->corpus->capacity; i++) {
        entry = data->corpus->entries[i];
        if(!entry.occupied) {
            continue;
        }

        p = entry.key;

        while(*p != '\0') {
            unsigned char current = (unsigned char)*p;
            if(!seen_chars[current]) {          // if we didn't find the character
                seen_chars[current] = true;
                pico_vec_push(&alphabet, (void*)p);  // push to the alphabet vector
            }
            p++;  // Move to the next character
        }
    }

    pico_vec_sort_chars(&alphabet);  // sort that alphavet vector pls

    struct PicoVec special_tokens;
    pico_vec_init(&special_tokens, 5);
    pico_vec_push(&special_tokens, "<|endoftext|>");
    pico_vec_push(&special_tokens, "<|unk|>");
    pico_vec_push(&special_tokens, "<|pad|>");
    pico_vec_push(&special_tokens, "<|bos|>");
    pico_vec_push(&special_tokens, "<|eos|>");

    struct PicoVec* next_vocab = pico_vec_append(&special_tokens, &alphabet);
    if(next_vocab != NULL) {
        pico_vec_free(data->vocab);
        free(data->vocab);
        data->vocab = next_vocab;
    }

    pico_vec_free(&special_tokens);
    pico_vec_free(&alphabet);
}


static inline void bpe_train(struct Tokenizer* tokenizer) {
    bpe_train_vocab(tokenizer); // train the vocab
}

// INFO: unlike the wordbased tokenizer, this tk is gonna initiate a tokenizer then have another function to fill in the
// tokens, so this function will be to return a tokenizer, another function will be used to fill up the corpus and ids
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
    data->corpus = pico_hashmap_init();
    data->vocab = malloc(sizeof(struct PicoVec));
    if(data->corpus == NULL || data->vocab == NULL) {
        if(data->corpus != NULL) {
            pico_hashmap_free(data->corpus);
        }
        free(data->vocab);
        return NULL;
    }

    pico_vec_init(data->vocab, 100);
    data->max_vocab_capacity = MAX_BPE_VOCAB_CAPACITY;

    tokenizer->ctx = context;
    tokenizer->methods = &BPE_TK_METHODS;
    tokenizer->data = data;

    return tokenizer;
}
