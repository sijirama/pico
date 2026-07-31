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
    pico_hashmap_free(data->vocab);
    data->vocab = NULL;
}

static size_t bpe_freq(struct Tokenizer* tokenizer, const char* token) {
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    return (size_t)(uintptr_t)pico_hashmap_get(data->vocab, token);
}

UTEST(bpe_tk, init_sets_context_methods_and_vocab) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);

    ASSERT_TRUE(tokenizer != NULL);
    ASSERT_TRUE(tokenizer->ctx == &ctx);
    ASSERT_TRUE(tokenizer->methods == &BPE_TK_METHODS);
    ASSERT_TRUE(tokenizer->data != NULL);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_TRUE(data->vocab != NULL);
    ASSERT_EQ(data->vocab->size, (size_t)0);
    ASSERT_EQ(data->vocab->capacity, (size_t)PICO_HASHMAP_INITIAL_CAPACITY);

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
    ASSERT_TRUE(pico_hashmap_contains(data->vocab, "pico"));
    ASSERT_TRUE(pico_hashmap_contains(data->vocab, "learns"));
    ASSERT_TRUE(pico_hashmap_contains(data->vocab, "fast"));
    ASSERT_FALSE(pico_hashmap_contains(data->vocab, "Pico"));
    ASSERT_EQ(data->vocab->size, (size_t)3);

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
    ASSERT_EQ(data->vocab->size, (size_t)0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}
