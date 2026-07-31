/*
 * Tests for the early BPE tokenizer pieces.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include <stdint.h>

#include "tokens/bpe-tk.h"
#include "utest.h"

static void free_bpe_map(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    pico_hashmap_free(data->corpus);
    if(data->vocab != NULL) {
        pico_vec_free(data->vocab);
        free(data->vocab);
    }
    data->corpus = NULL;
    data->vocab = NULL;
}

static size_t bpe_freq(struct Tokenizer* tokenizer, const char* token) {
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    return (size_t)(uintptr_t)pico_hashmap_get(data->corpus, token);
}

UTEST(bpe_tk, init_sets_context_methods_and_vocab) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);

    ASSERT_TRUE(tokenizer != NULL);
    ASSERT_TRUE(tokenizer->ctx == &ctx);
    ASSERT_TRUE(tokenizer->methods == &BPE_TK_METHODS);
    ASSERT_TRUE(tokenizer->data != NULL);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_TRUE(data->corpus != NULL);
    ASSERT_TRUE(data->vocab != NULL);
    ASSERT_EQ(data->corpus->size, (size_t)0);
    ASSERT_EQ(data->corpus->capacity, (size_t)PICO_HASHMAP_INITIAL_CAPACITY);
    ASSERT_EQ(data->vocab->size, (size_t)0);
    ASSERT_EQ(data->vocab->capacity, (size_t)100);
    ASSERT_EQ(data->max_vocab_capacity, MAX_BPE_VOCAB_CAPACITY);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, init_rejects_null_context) {
    ASSERT_TRUE(pico_bpe_tk_init(NULL) == NULL);
}

UTEST(bpe_tk, ingest_lowercases_and_splits_words) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    char text[] = "Pico, learns! FAST.";

    bpe_ingest_text(tokenizer, text);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, "pico"));
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, "learns"));
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, "fast"));
    ASSERT_FALSE(pico_hashmap_contains(data->corpus, "Pico"));
    ASSERT_EQ(data->corpus->size, (size_t)3);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, ingest_tracks_token_frequency) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    char first[] = "pico pico tensor";
    char second[] = "tensor pico";

    bpe_ingest_text(tokenizer, first);
    bpe_ingest_text(tokenizer, second);

    ASSERT_EQ(bpe_freq(tokenizer, "pico"), (size_t)3);
    ASSERT_EQ(bpe_freq(tokenizer, "tensor"), (size_t)2);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, ingest_rejects_invalid_inputs) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);

    bpe_ingest_text(NULL, NULL);
    bpe_ingest_text(tokenizer, NULL);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_EQ(data->corpus->size, (size_t)0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, train_builds_special_tokens_and_sorted_alphabet) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    char text[] = "cab ba";

    bpe_ingest_text(tokenizer, text);
    bpe_train_vocab(tokenizer);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_EQ(data->vocab->size, (size_t)8);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[0], "<|endoftext|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[1], "<|unk|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[2], "<|pad|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[3], "<|bos|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[4], "<|eos|>") == 0);
    ASSERT_TRUE(*(char*)data->vocab->data[5] == 'a');
    ASSERT_TRUE(*(char*)data->vocab->data[6] == 'b');
    ASSERT_TRUE(*(char*)data->vocab->data[7] == 'c');

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, train_empty_corpus_keeps_only_special_tokens) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);

    bpe_train_vocab(tokenizer);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_EQ(data->vocab->size, (size_t)5);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[0], "<|endoftext|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[4], "<|eos|>") == 0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}
