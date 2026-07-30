/*
 * Tests for the word-based tokenizer.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include <string.h>

#include "tokens/wordbased-tk.h"
#include "utest.h"

static void free_wordbased_map(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    pico_hashmap_free(data->word_to_id_map);
    data->word_to_id_map = NULL;
}

UTEST(wordbased_tk, init_adds_unk_token) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);

    ASSERT_TRUE(tokenizer != NULL);
    ASSERT_EQ(tokenizer->methods->len(tokenizer), (size_t)1);

    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    ASSERT_EQ(data->unk_token_id, (size_t)0);
    ASSERT_TRUE(pico_hashmap_contains(data->word_to_id_map, "<UNK>"));

    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(wordbased_tk, add_word_assigns_ids_once) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);

    ASSERT_TRUE(pico_wordbased_add_word(tokenizer, "hello"));
    ASSERT_TRUE(pico_wordbased_add_word(tokenizer, "world"));
    ASSERT_TRUE(pico_wordbased_add_word(tokenizer, "hello"));

    ASSERT_EQ(tokenizer->methods->len(tokenizer), (size_t)3);

    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    ASSERT_TRUE(*(size_t*)pico_hashmap_get(data->word_to_id_map, "hello") == 1);
    ASSERT_TRUE(*(size_t*)pico_hashmap_get(data->word_to_id_map, "world") == 2);

    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(wordbased_tk, encode_uses_vocab_and_unknown) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);

    pico_wordbased_add_word(tokenizer, "hello");
    pico_wordbased_add_word(tokenizer, "world");

    size_t* ids = tokenizer->methods->encode(tokenizer, "hello missing world");

    ASSERT_TRUE(ids != NULL);
    ASSERT_EQ(ids[0], (size_t)1);
    ASSERT_EQ(ids[1], (size_t)0);
    ASSERT_EQ(ids[2], (size_t)2);
    ASSERT_EQ(ids[3], (size_t)-1);

    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(wordbased_tk, decode_ids_to_words) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);

    pico_wordbased_add_word(tokenizer, "hello");
    pico_wordbased_add_word(tokenizer, "world");

    size_t ids[] = {1, 2, (size_t)-1};
    char* text = tokenizer->methods->decode(tokenizer, ids);

    ASSERT_TRUE(text != NULL);
    ASSERT_TRUE(strcmp(text, "hello world ") == 0);

    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
}

UTEST(wordbased_tk, rejects_invalid_inputs) {
    struct PicoContext ctx = pico_context_init();
    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);

    ASSERT_TRUE(pico_wordbased_create_init(NULL) == NULL);
    ASSERT_FALSE(pico_wordbased_add_word(NULL, "x"));
    ASSERT_FALSE(pico_wordbased_add_word(tokenizer, NULL));
    ASSERT_TRUE(tokenizer->methods->encode(NULL, "hello") == NULL);
    ASSERT_TRUE(tokenizer->methods->encode(tokenizer, NULL) == NULL);
    ASSERT_TRUE(tokenizer->methods->decode(NULL, (size_t[]){0, (size_t)-1}) == NULL);
    ASSERT_TRUE(tokenizer->methods->decode(tokenizer, NULL) == NULL);

    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
}
