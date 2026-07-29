
#pragma once

#include "arena.h"
#include "ctx.h"
#include "tokens/tokenizer.h"

struct WordBasedPicoTKData {
    void* word_to_id_map;           // HashMap or array mapping: const char* (word) -> size_t (ID)
    const char** id_to_word_array;  // Array mapping: size_t (ID) -> const char* (word)
    size_t unk_token_id;            // For out-of-vocabulary words
    size_t vocab_size;
};

size_t pico_wordbased_len(const struct Tokenizer* self) {
    if(!self || !self->data)
        return 0;
    const struct WordBasedPicoTKData* data = (const struct WordBasedPicoTKData*)self->data;
    return data->vocab_size;
}

void* pico_wordbased_encode(const struct Tokenizer* self, const char* text) {
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

        // Find ID from your vocabulary map
        size_t current_id = data->unk_token_id;

        // --- Pseudo Map Lookup Block ---
        // Replace this with your project's hash map lookup function:
        // if (hashmap_contains(data->word_to_id_map, token)) {
        //     current_id = hashmap_get(data->word_to_id_map, token);
        // }
        // -------------------------------

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
        token_ids = arena_alloc(context->arena, (count + 1) * sizeof(size_t));
    }
    token_ids[count] = (size_t)-1;  // Using -1 as a sentinel block ending indicator

    return (void*)token_ids;
}

void* pico_wordbased_decode(const struct Tokenizer* self, const float* ids) {
    if(!self || !ids)
        return NULL;
    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)self->data;
    struct PicoContext* context = self->ctx;

    // Initial string buffer setup
    size_t buffer_cap = 256;
    char* out_string = arena_alloc(context->arena, buffer_cap);
    out_string[0] = '\0';
    size_t current_len = 0;

    for(size_t i = 0; ids[i] != -1.0f; i++) {  // reading until our sentinel value
        size_t target_id = (size_t)ids[i];
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

struct Tokenizer* pico_wordbased_create_init(struct PicoContext* context) {
    struct Tokenizer* tokenizer = arena_alloc(context->arena, sizeof(struct Tokenizer));
    struct WordBasedPicoTKData* data = arena_alloc(context->arena, sizeof(struct WordBasedPicoTKData));

    // Initialize vocabulary constraints
    data->vocab_size = 0;
    data->unk_token_id = 0;
    data->word_to_id_map = NULL;    // Initialize your chosen map utility here
    data->id_to_word_array = NULL;  // Allocate your array memory blocks here

    tokenizer->ctx = context;
    tokenizer->methods = &WORDBASE_TK_METHODS;
    tokenizer->data = data;

    return tokenizer;
}
