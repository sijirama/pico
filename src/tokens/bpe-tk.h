/*

INFO: bpe tokenizer flow

you do not need to call all the helper functions in this file directly.
the normal flow is:

    struct Tokenizer* tk = pico_bpe_tk_init(ctx);
    bpe_ingest_text(tk, text);
    bpe_train(tk);
    size_t* ids = tk->methods->encode(tk, "hug bug");
    char* text = tk->methods->decode(tk, ids);

bpe_train handles the internal steps for you:

    train base vocab -> create word splits -> count pairs -> merge best pairs

example:

    corpus: "hug hug bug"
    base vocab: ["b", "g", "h", "u", ...]
    learned merge: ("u", "g") -> "ug"
    learned merge: ("h", "ug") -> "hug"
    encode "thug" -> ["t", "hug"] -> ids

this follows the simple hugging face bpe walkthrough:
https://huggingface.co/learn/llm-course/en/chapter6/5

the small bpe_* helpers below exist so the training loop is easier to read.
we normally only need init, ingest, train, encode, and decode.

*/

#pragma once

#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "lib/pico_map.h"
#include "lib/pico_vec_sort.h"
#include "lib/pico_vector.h"
#include "tokens/tokenizer.h"

#define MAX_BPE_VOCAB_CAPACITY 1280
#define BPE_PAIR_SEPARATOR '\t'
#define BPE_UNK_TOKEN_ID 1

struct BPEMergeRule {
    char* left;
    char* right;
    char* merged;
};

// INFO: bpe training has 2 main bits of state here
// corpus is word -> frequency, vocab is the actual token list we are learning
struct BPEPicoTKData {
    struct PicoHashMap* corpus;  // char * | size_t
    struct PicoVec* vocab;
    struct PicoHashMap* token_to_id;
    struct PicoVec* merges;
    size_t* id_values;
    size_t vocab_id_capacity;
    int max_vocab_capacity;
};

static inline size_t pico_bpe_tk_len(const struct Tokenizer* self) {
    if(!self || !self->data)
        return 0;
    const struct BPEPicoTKData* data = (const struct BPEPicoTKData*)self->data;
    return data->vocab == NULL ? 0 : data->vocab->size;
}

// ===================== these are util functions for the bpe to operate

// INFO: basic normalization for now, later this can become its own tokenizer step
static inline void convert_to_lowercase(char* str) {
    // Loop until the null-terminator '\0' is reached
    while(*str != '\0') {
        *str = tolower((unsigned char)*str);  // Convert current char
        str++;                                // Move pointer to next char
    }
}

// INFO: this is the first rough pre-tokenizer
// it turns raw text into word-like chunks before bpe starts splitting words into chars
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

// INFO: bpe split tokens are temporary training objects, so these helpers use heap strings
// the final vocab tokens live in the context arena because the tokenizer owns them for longer
static inline char* bpe_heap_strdup(const char* text) {
    size_t len = strlen(text);
    char* copy = malloc(len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static inline char* bpe_arena_strdup(struct PicoContext* ctx, const char* text) {
    size_t len = strlen(text);
    char* copy = arena_alloc(ctx->arena, len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static inline char* bpe_heap_char_token(char c) {
    char* token = malloc(2);
    if(token == NULL) {
        return NULL;
    }

    token[0] = c;
    token[1] = '\0';
    return token;
}

// INFO: when include_separator is true this creates a hashmap key like "a\tb"
// we need the separator because a pair is 2 tokens, not just one merged string yet
static inline char* bpe_join_tokens_heap(const char* a, const char* b, bool include_separator) {
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    size_t separator_len = include_separator ? 1 : 0;
    char* joined = malloc(a_len + separator_len + b_len + 1);
    if(joined == NULL) {
        return NULL;
    }

    memcpy(joined, a, a_len);
    if(include_separator) {
        joined[a_len] = BPE_PAIR_SEPARATOR;
    }
    memcpy(joined + a_len + separator_len, b, b_len + 1);
    return joined;
}

// INFO: this is for the new vocab token after a merge, so "a" + "b" becomes "ab"
static inline char* bpe_join_tokens_arena(struct PicoContext* ctx, const char* a, const char* b) {
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    char* joined = arena_alloc(ctx->arena, a_len + b_len + 1);
    if(joined == NULL) {
        return NULL;
    }

    memcpy(joined, a, a_len);
    memcpy(joined + a_len, b, b_len + 1);
    return joined;
}

// INFO: pair keys are stored as one string in the map, so this cracks "a\tb" back into a and b
// this mutates the copy passed into it by replacing the separator with a null terminator
static inline bool bpe_split_pair_key(char* pair_key, const char** left, const char** right) {
    char* separator = strchr(pair_key, BPE_PAIR_SEPARATOR);
    if(separator == NULL) {
        return false;
    }

    *separator = '\0';
    *left = pair_key;
    *right = separator + 1;
    return true;
}

// INFO: ids are stored as pointers in the map, so this array gives each id a stable address
static inline bool bpe_register_vocab_token(struct BPEPicoTKData* data, const char* token, size_t id) {
    if(data == NULL || data->token_to_id == NULL || data->id_values == NULL || token == NULL ||
       id >= data->vocab_id_capacity) {
        return false;
    }

    data->id_values[id] = id;
    return pico_hashmap_insert(data->token_to_id, token, &data->id_values[id]);
}

static inline bool bpe_rebuild_token_to_id(struct BPEPicoTKData* data) {
    if(data == NULL || data->vocab == NULL || data->id_values == NULL) {
        return false;
    }

    if(data->token_to_id != NULL) {
        pico_hashmap_free(data->token_to_id);
    }

    data->token_to_id = pico_hashmap_init_with_capacity(data->vocab->capacity > 16 ? data->vocab->capacity : 16);
    if(data->token_to_id == NULL) {
        return false;
    }

    for(size_t i = 0; i < data->vocab->size; i++) {
        if(!bpe_register_vocab_token(data, (char*)data->vocab->data[i], i)) {
            return false;
        }
    }

    return true;
}

static inline bool bpe_store_merge_rule(struct Tokenizer* tokenizer, const char* left, const char* right,
                                        const char* merged) {
    if(tokenizer == NULL || tokenizer->ctx == NULL || tokenizer->data == NULL || left == NULL || right == NULL ||
       merged == NULL) {
        return false;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    if(data->merges == NULL) {
        return false;
    }

    struct BPEMergeRule* rule = arena_alloc(tokenizer->ctx->arena, sizeof(struct BPEMergeRule));
    if(rule == NULL) {
        return false;
    }

    rule->left = bpe_arena_strdup(tokenizer->ctx, left);
    rule->right = bpe_arena_strdup(tokenizer->ctx, right);
    rule->merged = bpe_arena_strdup(tokenizer->ctx, merged);
    if(rule->left == NULL || rule->right == NULL || rule->merged == NULL) {
        return false;
    }

    pico_vec_push(data->merges, rule);
    return true;
}

// INFO: a split vec owns each string inside it, so freeing the vec also means freeing every token
static inline void bpe_free_split_vec(struct PicoVec* split) {
    if(split == NULL) {
        return;
    }

    for(size_t i = 0; i < split->size; i++) {
        free(split->data[i]);
    }

    pico_vec_free(split);
    free(split);
}

// INFO: splits is word -> split vec, so cleanup has to walk the hashmap values first
static inline void bpe_free_splits(struct PicoHashMap* splits) {
    if(splits == NULL) {
        return;
    }

    for(size_t i = 0; i < splits->capacity; i++) {
        if(splits->entries[i].occupied) {
            bpe_free_split_vec((struct PicoVec*)splits->entries[i].value);
        }
    }

    pico_hashmap_free(splits);
}

static inline struct PicoVec* bpe_create_word_split(const char* word) {
    if(word == NULL) {
        return NULL;
    }

    struct PicoVec* split = malloc(sizeof(struct PicoVec));
    if(split == NULL) {
        return NULL;
    }

    size_t word_len = strlen(word);
    pico_vec_init(split, word_len > 0 ? word_len : 1);

    for(size_t i = 0; i < word_len; i++) {
        char* token = bpe_heap_char_token(word[i]);
        if(token != NULL) {
            pico_vec_push(split, token);
        }
    }

    return split;
}

// INFO: apply one merge to one split vec
// this is the core encode/training move: scan left to right and replace matching neighbors
static inline bool bpe_apply_merge_to_split(struct PicoVec* split, const char* left, const char* right) {
    if(split == NULL || left == NULL || right == NULL || split->size < 2) {
        return true;
    }

    struct PicoVec next_split;
    pico_vec_init(&next_split, split->size);

    size_t i = 0;
    while(i < split->size) {
        if(i + 1 < split->size && strcmp((char*)split->data[i], left) == 0 &&
           strcmp((char*)split->data[i + 1], right) == 0) {
            char* merged = bpe_join_tokens_heap((char*)split->data[i], (char*)split->data[i + 1], false);
            free(split->data[i]);
            free(split->data[i + 1]);
            if(merged == NULL) {
                pico_vec_free(&next_split);
                return false;
            }
            pico_vec_push(&next_split, merged);
            i += 2;
        } else {
            pico_vec_push(&next_split, split->data[i]);
            i++;
        }
    }

    pico_vec_free(split);
    *split = next_split;
    return true;
}

// INFO: pair_freqs is pair -> count
// word_freq matters because a repeated word should make all its pairs count more
static inline void bpe_pair_freq_increment(struct PicoHashMap* pair_freqs, const char* left, const char* right,
                                           size_t word_freq) {
    char* pair_key = bpe_join_tokens_heap(left, right, true);
    if(pair_key == NULL) {
        return;
    }

    struct PicoHashEntry* entry = pico_hashmap_find_entry(pair_freqs, pair_key);
    if(entry == NULL) {
        pico_hashmap_insert(pair_freqs, pair_key, (void*)(uintptr_t)word_freq);
    } else {
        size_t next_freq = (size_t)(uintptr_t)entry->value + word_freq;
        entry->value = (void*)(uintptr_t)next_freq;
    }

    free(pair_key);
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

// train the initial vocab and add special tokens too
// INFO: before learning merges, bpe starts with every character it has seen in the corpus
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
            if(!seen_chars[current]) {  // if we didn't find the character
                seen_chars[current] = true;
                char* token = bpe_arena_strdup(tokenizer->ctx, (char[]){*p, '\0'});
                if(token != NULL) {
                    pico_vec_push(&alphabet, token);  // push to the alphabet vector
                }
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
        bpe_rebuild_token_to_id(data);
    }

    pico_vec_free(&special_tokens);
    pico_vec_free(&alphabet);
}

// INFO: this creates the python-blog style splits map
// example: "hug" -> ["h", "u", "g"], and later merges will rewrite that vec
static inline void bpe_create_splits(struct PicoHashMap* splits, struct PicoHashMap* corpus) {
    if(splits == NULL || corpus == NULL) {
        return;
    }

    for(size_t i = 0; i < corpus->capacity; i++) {
        struct PicoHashEntry entry = corpus->entries[i];
        if(!entry.occupied) {
            continue;
        }

        struct PicoVec* split = bpe_create_word_split(entry.key);
        if(split == NULL) {
            return;
        }

        pico_hashmap_insert(splits, entry.key, split);
    }
}

// INFO: this is the scoring step for each bpe round
// it walks every current split and counts adjacent pairs like ("h", "u") or ("u", "g")
static inline void bpe_compute_pair_freqs(struct PicoHashMap* pair_freqs, struct PicoHashMap* splits,
                                          struct PicoHashMap* corpus) {
    if(pair_freqs == NULL || splits == NULL || corpus == NULL) {
        return;
    }

    for(size_t i = 0; i < splits->capacity; i++) {
        struct PicoHashEntry split_entry = splits->entries[i];
        if(!split_entry.occupied) {
            continue;
        }

        struct PicoVec* split = (struct PicoVec*)split_entry.value;
        if(split == NULL || split->size < 2) {
            continue;
        }

        size_t word_freq = (size_t)(uintptr_t)pico_hashmap_get(corpus, split_entry.key);
        for(size_t j = 0; j + 1 < split->size; j++) {
            bpe_pair_freq_increment(pair_freqs, (char*)split->data[j], (char*)split->data[j + 1], word_freq);
        }
    }
}

// INFO: choose the pair with the biggest count for the next merge rule
// ties just fall to whichever pair the hashmap exposes first, which is fine for this simple version
static inline char* bpe_best_pair_key(struct PicoHashMap* pair_freqs) {
    if(pair_freqs == NULL || pair_freqs->size == 0) {
        return NULL;
    }

    char* best_pair = NULL;
    size_t max_freq = 0;
    for(size_t i = 0; i < pair_freqs->capacity; i++) {
        struct PicoHashEntry entry = pair_freqs->entries[i];
        if(!entry.occupied) {
            continue;
        }

        size_t freq = (size_t)(uintptr_t)entry.value;
        if(best_pair == NULL || freq > max_freq) {
            best_pair = entry.key;
            max_freq = freq;
        }
    }

    return best_pair;
}

// INFO: apply one learned merge to every word split
// example: if the best pair is "u" + "g", then ["h", "u", "g"] becomes ["h", "ug"]
static inline void bpe_merge_pair(struct PicoHashMap* splits, const char* pair_key) {
    if(splits == NULL || pair_key == NULL) {
        return;
    }

    char* pair_copy = bpe_heap_strdup(pair_key);
    if(pair_copy == NULL) {
        return;
    }

    const char* left = NULL;
    const char* right = NULL;
    if(!bpe_split_pair_key(pair_copy, &left, &right)) {
        free(pair_copy);
        return;
    }

    for(size_t i = 0; i < splits->capacity; i++) {
        struct PicoHashEntry* entry = &splits->entries[i];
        if(!entry->occupied) {
            continue;
        }

        bpe_apply_merge_to_split((struct PicoVec*)entry->value, left, right);
    }

    free(pair_copy);
}

// INFO: full bpe training loop
// 1. build the starting vocab from special tokens + characters
// 2. split every corpus word into characters
// 3. keep merging the most frequent adjacent pair until the vocab is full
static inline void bpe_train(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    bpe_train_vocab(tokenizer);  // train the initial vocab
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;

    struct PicoHashMap* splits = pico_hashmap_init();  // word: ["w","o","r","d"]
    if(splits == NULL) {
        return;
    }
    bpe_create_splits(splits, data->corpus);

    while(data->vocab->size < data->max_vocab_capacity) {
        struct PicoHashMap* pair_freqs = pico_hashmap_init();
        if(pair_freqs == NULL) {
            break;
        }

        bpe_compute_pair_freqs(pair_freqs, splits, data->corpus);
        char* best_pair = bpe_best_pair_key(pair_freqs);
        if(best_pair == NULL) {
            pico_hashmap_free(pair_freqs);
            break;
        }

        char* pair_copy = bpe_heap_strdup(best_pair);
        const char* left = NULL;
        const char* right = NULL;
        if(pair_copy == NULL || !bpe_split_pair_key(pair_copy, &left, &right)) {
            free(pair_copy);
            pico_hashmap_free(pair_freqs);
            break;
        }

        char* merged_vocab_token = bpe_join_tokens_arena(tokenizer->ctx, left, right);
        bpe_merge_pair(splits, best_pair);
        if(merged_vocab_token != NULL) {
            size_t next_id = data->vocab->size;
            pico_vec_push(data->vocab, merged_vocab_token);
            bpe_register_vocab_token(data, merged_vocab_token, next_id);
            bpe_store_merge_rule(tokenizer, left, right, merged_vocab_token);
        }

        free(pair_copy);
        pico_hashmap_free(pair_freqs);
    }

    bpe_free_splits(splits);
}

static inline void* pico_bpe_tk_encode(const struct Tokenizer* self, const char* text) {
    if(self == NULL || self->ctx == NULL || self->data == NULL || text == NULL) {
        return NULL;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)self->data;
    if(data->token_to_id == NULL || data->merges == NULL) {
        return NULL;
    }

    char* normalized = bpe_heap_strdup(text);
    if(normalized == NULL) {
        return NULL;
    }

    convert_to_lowercase(normalized);

    struct PicoVec words;
    pico_vec_init(&words, 16);
    bpe_pretokenize(&words, normalized);

    struct PicoVec splits;
    pico_vec_init(&splits, words.size > 0 ? words.size : 1);

    for(size_t i = 0; i < words.size; i++) {
        struct PicoVec* split = bpe_create_word_split((char*)words.data[i]);
        if(split != NULL) {
            pico_vec_push(&splits, split);
        }
        free(words.data[i]);
    }

    pico_vec_free(&words);

    for(size_t merge_i = 0; merge_i < data->merges->size; merge_i++) {
        struct BPEMergeRule* rule = (struct BPEMergeRule*)data->merges->data[merge_i];
        for(size_t split_i = 0; split_i < splits.size; split_i++) {
            bpe_apply_merge_to_split((struct PicoVec*)splits.data[split_i], rule->left, rule->right);
        }
    }

    // INFO: bpe merges only shrink splits, so original text length is a safe upper bound for ids
    size_t id_capacity = strlen(normalized) + 1;
    size_t* ids = arena_alloc(self->ctx->arena, sizeof(size_t) * (id_capacity > 0 ? id_capacity : 1));
    if(ids == NULL) {
        for(size_t i = 0; i < splits.size; i++) {
            bpe_free_split_vec((struct PicoVec*)splits.data[i]);
        }
        pico_vec_free(&splits);
        free(normalized);
        return NULL;
    }

    size_t count = 0;
    for(size_t split_i = 0; split_i < splits.size; split_i++) {
        struct PicoVec* split = (struct PicoVec*)splits.data[split_i];
        for(size_t token_i = 0; token_i < split->size; token_i++) {
            size_t* found_id = (size_t*)pico_hashmap_get(data->token_to_id, (char*)split->data[token_i]);
            ids[count++] = found_id == NULL ? BPE_UNK_TOKEN_ID : *found_id;
        }
    }

    ids[count] = (size_t)-1;

    for(size_t i = 0; i < splits.size; i++) {
        bpe_free_split_vec((struct PicoVec*)splits.data[i]);
    }
    pico_vec_free(&splits);
    free(normalized);
    return ids;
}

static inline void* pico_bpe_tk_decode(const struct Tokenizer* self, const size_t* ids) {
    if(self == NULL || self->ctx == NULL || self->data == NULL || ids == NULL) {
        return NULL;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)self->data;
    if(data->vocab == NULL) {
        return NULL;
    }

    size_t buffer_cap = 128;
    char* out = arena_alloc(self->ctx->arena, buffer_cap);
    if(out == NULL) {
        return NULL;
    }

    out[0] = '\0';
    size_t out_len = 0;
    for(size_t i = 0; ids[i] != (size_t)-1; i++) {
        size_t id = ids[i];
        const char* token = id < data->vocab->size ? (const char*)data->vocab->data[id] : "<|unk|>";
        size_t token_len = strlen(token);

        if(out_len + token_len + 1 > buffer_cap) {
            buffer_cap = (buffer_cap + token_len + 1) * 2;
            char* next_out = arena_alloc(self->ctx->arena, buffer_cap);
            if(next_out == NULL) {
                return NULL;
            }
            memcpy(next_out, out, out_len + 1);
            out = next_out;
        }

        memcpy(out + out_len, token, token_len + 1);
        out_len += token_len;
    }

    return out;
}

static const struct TokenizerVTable BPE_TK_METHODS = {
    .len = pico_bpe_tk_len, .encode = pico_bpe_tk_encode, .decode = pico_bpe_tk_decode};

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
    data->token_to_id = pico_hashmap_init_with_capacity(MAX_BPE_VOCAB_CAPACITY);
    data->vocab = malloc(sizeof(struct PicoVec));
    data->merges = malloc(sizeof(struct PicoVec));
    data->id_values = arena_alloc(context->arena, sizeof(size_t) * MAX_BPE_VOCAB_CAPACITY);
    data->vocab_id_capacity = MAX_BPE_VOCAB_CAPACITY;
    if(data->corpus == NULL || data->token_to_id == NULL || data->vocab == NULL || data->merges == NULL ||
       data->id_values == NULL) {
        if(data->corpus != NULL) {
            pico_hashmap_free(data->corpus);
        }
        if(data->token_to_id != NULL) {
            pico_hashmap_free(data->token_to_id);
        }
        free(data->vocab);
        free(data->merges);
        return NULL;
    }

    pico_vec_init(data->vocab, 100);
    pico_vec_init(data->merges, 100);
    data->max_vocab_capacity = MAX_BPE_VOCAB_CAPACITY;

    tokenizer->ctx = context;
    tokenizer->methods = &BPE_TK_METHODS;
    tokenizer->data = data;

    return tokenizer;
}
