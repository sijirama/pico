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
    pico_hashmap_free(data->token_to_id);
    if(data->vocab != NULL) {
        pico_vec_free(data->vocab);
        free(data->vocab);
    }
    if(data->merges != NULL) {
        pico_vec_free(data->merges);
        free(data->merges);
    }
    data->corpus = NULL;
    data->vocab = NULL;
    data->token_to_id = NULL;
    data->merges = NULL;
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
    ASSERT_TRUE(data->token_to_id != NULL);
    ASSERT_TRUE(data->merges != NULL);
    ASSERT_TRUE(data->id_values != NULL);
    ASSERT_EQ(data->corpus->size, (size_t)0);
    ASSERT_EQ(data->corpus->capacity, (size_t)PICO_HASHMAP_INITIAL_CAPACITY);
    ASSERT_EQ(data->vocab->size, (size_t)0);
    ASSERT_EQ(data->vocab->capacity, (size_t)100);
    ASSERT_EQ(data->token_to_id->size, (size_t)0);
    ASSERT_EQ(data->merges->size, (size_t)0);
    ASSERT_EQ(data->vocab_id_capacity, (size_t)MAX_BPE_VOCAB_CAPACITY);
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
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, BPE_SPACE_TOKEN));
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, "learns"));
    ASSERT_TRUE(pico_hashmap_contains(data->corpus, "fast"));
    ASSERT_FALSE(pico_hashmap_contains(data->corpus, "Pico"));
    ASSERT_EQ(data->corpus->size, (size_t)4);
    ASSERT_EQ(bpe_freq(tokenizer, BPE_SPACE_TOKEN), (size_t)2);

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
    ASSERT_EQ(data->vocab->size, (size_t)9);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[0], "<|endoftext|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[1], "<|unk|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[2], "<|pad|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[3], "<|bos|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[4], "<|eos|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[5], BPE_SPACE_TOKEN) == 0);
    ASSERT_TRUE(*(char*)data->vocab->data[6] == 'a');
    ASSERT_TRUE(*(char*)data->vocab->data[7] == 'b');
    ASSERT_TRUE(*(char*)data->vocab->data[8] == 'c');

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, train_empty_corpus_keeps_only_special_tokens) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);

    bpe_train_vocab(tokenizer);

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    ASSERT_EQ(data->vocab->size, (size_t)6);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[0], "<|endoftext|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[4], "<|eos|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[5], BPE_SPACE_TOKEN) == 0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, train_learns_most_frequent_merges_until_capacity) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    char text[] = "hug hug bug";

    data->max_vocab_capacity = 12;
    bpe_ingest_text(tokenizer, text);
    bpe_train(tokenizer);

    ASSERT_EQ(data->vocab->size, (size_t)12);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[10], "ug") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[11], "hug") == 0);
    ASSERT_EQ(data->merges->size, (size_t)2);

    struct BPEMergeRule* first_merge = (struct BPEMergeRule*)data->merges->data[0];
    struct BPEMergeRule* second_merge = (struct BPEMergeRule*)data->merges->data[1];
    ASSERT_TRUE(strcmp(first_merge->left, "u") == 0);
    ASSERT_TRUE(strcmp(first_merge->right, "g") == 0);
    ASSERT_TRUE(strcmp(first_merge->merged, "ug") == 0);
    ASSERT_TRUE(strcmp(second_merge->left, "h") == 0);
    ASSERT_TRUE(strcmp(second_merge->right, "ug") == 0);
    ASSERT_TRUE(strcmp(second_merge->merged, "hug") == 0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, train_stops_when_no_pairs_are_left) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    char text[] = "a";

    data->max_vocab_capacity = 20;
    bpe_ingest_text(tokenizer, text);
    bpe_train(tokenizer);

    ASSERT_EQ(data->vocab->size, (size_t)7);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[0], "<|endoftext|>") == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[5], BPE_SPACE_TOKEN) == 0);
    ASSERT_TRUE(strcmp((char*)data->vocab->data[6], "a") == 0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, encode_replays_learned_merges_in_order) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    char train_text[] = "hug hug bug";

    data->max_vocab_capacity = 12;
    bpe_ingest_text(tokenizer, train_text);
    bpe_train(tokenizer);

    size_t* ids = (size_t*)tokenizer->methods->encode(tokenizer, "thug bug");

    ASSERT_TRUE(ids != NULL);
    ASSERT_EQ(ids[0], (size_t)BPE_UNK_TOKEN_ID);
    ASSERT_EQ(ids[1], (size_t)11);
    ASSERT_EQ(ids[2], (size_t)5);
    ASSERT_EQ(ids[3], (size_t)6);
    ASSERT_EQ(ids[4], (size_t)10);
    ASSERT_EQ(ids[5], (size_t)-1);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(bpe_tk, decode_maps_ids_back_to_tokens) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    char train_text[] = "hug hug bug";

    data->max_vocab_capacity = 12;
    bpe_ingest_text(tokenizer, train_text);
    bpe_train(tokenizer);

    size_t ids[] = {11, 5, 6, 10, (size_t)-1};
    char* decoded = (char*)tokenizer->methods->decode(tokenizer, ids);

    ASSERT_TRUE(decoded != NULL);
    ASSERT_TRUE(strcmp(decoded, "hug bug") == 0);

    free_bpe_map(tokenizer);
    pico_context_destroy(&ctx);
}
