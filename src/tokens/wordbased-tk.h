
#pragma once

#include <stdbool.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "lib/pico_map.h"
#include "tokens/tokenizer.h"

#define PICO_WORDBASED_INITIAL_VOCAB_CAPACITY 1024

struct WordBasedPicoTKData {
    struct PicoHashMap* word_to_id_map;
    size_t* id_values;
    const char** id_to_word_array;
    size_t unk_token_id;            // For out-of-vocabulary words
    size_t vocab_size;
    size_t vocab_capacity;
};

static inline size_t pico_wordbased_len(const struct Tokenizer* self) {
    if(!self || !self->data)
        return 0;
    const struct WordBasedPicoTKData* data = (const struct WordBasedPicoTKData*)self->data;
    return data->vocab_size;
}

static inline void* pico_wordbased_encode(const struct Tokenizer* self, const char* text) {
    if(!self || !text)
        return NULL;
    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)self->data;
    struct PicoContext* context = self->ctx;

    // Temporary dynamic array to hold collected IDs
    size_t capacity = 32;
    size_t count = 0;
    size_t* token_ids = arena_alloc(context->arena, capacity * sizeof(size_t));

    // Make a mutable copy of the string to safely use strtok
    size_t text_len = strlen(text);
    char* text_copy = arena_alloc(context->arena, text_len + 1);
    strcpy(text_copy, text);

    // Naive whitespace tokenization loop
    char* token = strtok(text_copy, " \t\r\n");
    while(token != NULL) {
        // Optional: Strip trailing/leading punctuation from 'token' here if needed

        size_t current_id = data->unk_token_id;
        size_t* found_id = (size_t*)pico_hashmap_get(data->word_to_id_map, token);
        if(found_id != NULL) {
            current_id = *found_id;
        }

        // Append ID to your array
        if(count >= capacity) {
            capacity *= 2;
            size_t* new_ids = arena_alloc(context->arena, capacity * sizeof(size_t));
            memcpy(new_ids, token_ids, count * sizeof(size_t));
            token_ids = new_ids;
        }
        token_ids[count++] = current_id;

        token = strtok(NULL, " \t\r\n");
    }

    // Null-terminate or store size. Let's prepend or append the size,
    // or let your architecture assume a special trailing sentinel ID (like 0xFFFFFFFF)
    if(count >= capacity) {
        size_t* new_ids = arena_alloc(context->arena, (count + 1) * sizeof(size_t));
        memcpy(new_ids, token_ids, count * sizeof(size_t));
        token_ids = new_ids;
    }
    token_ids[count] = (size_t)-1;  // Using -1 as a sentinel block ending indicator

    return (void*)token_ids;
}

static inline void* pico_wordbased_decode(const struct Tokenizer* self, const size_t* ids) {
    if(!self || !ids)
        return NULL;
    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)self->data;
    struct PicoContext* context = self->ctx;

    // Initial string buffer setup
    size_t buffer_cap = 256;
    char* out_string = arena_alloc(context->arena, buffer_cap);
    out_string[0] = '\0';
    size_t current_len = 0;

    for(size_t i = 0; ids[i] != (size_t)-1; i++) {  // reading until our sentinel value
        size_t target_id = ids[i];
        const char* word = " <UNK> ";

        if(target_id < data->vocab_size) {
            word = data->id_to_word_array[target_id];
        }

        size_t word_len = strlen(word);

        // Resize buffer if it risks spilling over
        if(current_len + word_len + 2 >= buffer_cap) {
            buffer_cap = (buffer_cap + word_len) * 2;
            char* new_buffer = arena_alloc(context->arena, buffer_cap);
            strcpy(new_buffer, out_string);
            out_string = new_buffer;
        }

        // Concatenate token string
        strcat(out_string, word);
        strcat(out_string, " ");  // Add space spacing between words
        current_len += (word_len + 1);
    }

    return (void*)out_string;
}

static const struct TokenizerVTable WORDBASE_TK_METHODS = {
    .len = pico_wordbased_len, .encode = pico_wordbased_encode, .decode = pico_wordbased_decode};

static bool pico_wordbased_grow_vocab(struct Tokenizer* tokenizer) {
    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    size_t next_capacity = data->vocab_capacity * 2;

    const char** next_words = arena_alloc(tokenizer->ctx->arena, sizeof(char*) * next_capacity);
    size_t* next_ids = arena_alloc(tokenizer->ctx->arena, sizeof(size_t) * next_capacity);
    if(next_words == NULL || next_ids == NULL) {
        return false;
    }

    memcpy(next_words, data->id_to_word_array, sizeof(char*) * data->vocab_size);
    memcpy(next_ids, data->id_values, sizeof(size_t) * data->vocab_size);

    data->id_to_word_array = next_words;
    data->id_values = next_ids;
    data->vocab_capacity = next_capacity;
    return true;
}

static char* pico_wordbased_copy_word(struct PicoContext* context, const char* word) {
    size_t len = strlen(word);
    char* copy = arena_alloc(context->arena, len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, word, len + 1);
    return copy;
}

static inline bool pico_wordbased_add_word(struct Tokenizer* tokenizer, const char* word) {
    if(tokenizer == NULL || tokenizer->ctx == NULL || tokenizer->ctx->arena == NULL || tokenizer->data == NULL ||
       word == NULL) {
        return false;
    }

    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    if(data->word_to_id_map == NULL) {
        return false;
    }

    if(pico_hashmap_contains(data->word_to_id_map, word)) {
        return true;
    }

    if(data->vocab_size >= data->vocab_capacity && !pico_wordbased_grow_vocab(tokenizer)) {
        return false;
    }

    size_t id = data->vocab_size;
    char* word_copy = pico_wordbased_copy_word(tokenizer->ctx, word);
    if(word_copy == NULL) {
        return false;
    }

    data->id_values[id] = id;
    data->id_to_word_array[id] = word_copy;
    data->vocab_size += 1;

    return pico_hashmap_insert(data->word_to_id_map, word_copy, &data->id_values[id]);
}

static inline struct Tokenizer* pico_wordbased_create_init(struct PicoContext* context) {
    if(context == NULL || context->arena == NULL) {
        return NULL;
    }

    struct Tokenizer* tokenizer = arena_alloc(context->arena, sizeof(struct Tokenizer));
    struct WordBasedPicoTKData* data = arena_alloc(context->arena, sizeof(struct WordBasedPicoTKData));
    if(tokenizer == NULL || data == NULL) {
        return NULL;
    }

    // Initialize vocabulary constraints
    data->vocab_size = 0;
    data->unk_token_id = 0;
    data->vocab_capacity = PICO_WORDBASED_INITIAL_VOCAB_CAPACITY;
    data->word_to_id_map = pico_hashmap_init_with_capacity(PICO_WORDBASED_INITIAL_VOCAB_CAPACITY);
    data->id_values = arena_alloc(context->arena, sizeof(size_t) * data->vocab_capacity);
    data->id_to_word_array = arena_alloc(context->arena, sizeof(char*) * data->vocab_capacity);
    if(data->word_to_id_map == NULL || data->id_values == NULL || data->id_to_word_array == NULL) {
        if(data->word_to_id_map != NULL) {
            pico_hashmap_free(data->word_to_id_map);
        }
        return NULL;
    }

    tokenizer->ctx = context;
    tokenizer->methods = &WORDBASE_TK_METHODS;
    tokenizer->data = data;

    if(!pico_wordbased_add_word(tokenizer, "<UNK>")) {
        pico_hashmap_free(data->word_to_id_map);
        return NULL;
    }

    return tokenizer;
}
