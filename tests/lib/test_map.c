/*
 * Tests for PicoHashMap.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "lib/pico_map.h"
#include "tensor.h"
#include "utest.h"

UTEST(hashmap, init_sets_fields) {
    struct PicoHashMap* map = pico_hashmap_init();

    ASSERT_TRUE(map != NULL);
    ASSERT_TRUE(map->entries != NULL);
    ASSERT_EQ(map->size, (size_t)0);
    ASSERT_EQ(map->capacity, (size_t)PICO_HASHMAP_INITIAL_CAPACITY);

    pico_hashmap_free(map);
}

UTEST(hashmap, insert_and_get_int_pointer) {
    struct PicoHashMap* map = pico_hashmap_init();
    int value = 42;

    ASSERT_TRUE(pico_hashmap_insert(map, "answer", &value));
    ASSERT_TRUE(pico_hashmap_get(map, "answer") == &value);
    ASSERT_TRUE(*(int*)pico_hashmap_get(map, "answer") == 42);

    pico_hashmap_free(map);
}

UTEST(hashmap, update_existing_key) {
    struct PicoHashMap* map = pico_hashmap_init();
    int old_value = 1;
    int new_value = 2;

    ASSERT_TRUE(pico_hashmap_insert(map, "id", &old_value));
    ASSERT_TRUE(pico_hashmap_insert(map, "id", &new_value));

    ASSERT_EQ(map->size, (size_t)1);
    ASSERT_TRUE(pico_hashmap_get(map, "id") == &new_value);

    pico_hashmap_free(map);
}

UTEST(hashmap, missing_key_returns_null) {
    struct PicoHashMap* map = pico_hashmap_init();
    int value = 7;

    ASSERT_TRUE(pico_hashmap_insert(map, "present", &value));
    ASSERT_TRUE(pico_hashmap_get(map, "missing") == NULL);
    ASSERT_FALSE(pico_hashmap_contains(map, "missing"));

    pico_hashmap_free(map);
}

UTEST(hashmap, contains_works_for_null_value) {
    struct PicoHashMap* map = pico_hashmap_init();

    ASSERT_TRUE(pico_hashmap_insert(map, "none", NULL));
    ASSERT_TRUE(pico_hashmap_get(map, "none") == NULL);
    ASSERT_TRUE(pico_hashmap_contains(map, "none"));

    pico_hashmap_free(map);
}

UTEST(hashmap, handles_collisions) {
    struct PicoHashMap* map = pico_hashmap_init();
    int a = 1;
    int q = 17;

    ASSERT_EQ(pico_hash_string("a") % map->capacity, pico_hash_string("q") % map->capacity);

    ASSERT_TRUE(pico_hashmap_insert(map, "a", &a));
    ASSERT_TRUE(pico_hashmap_insert(map, "q", &q));

    ASSERT_TRUE(*(int*)pico_hashmap_get(map, "a") == 1);
    ASSERT_TRUE(*(int*)pico_hashmap_get(map, "q") == 17);

    pico_hashmap_free(map);
}

UTEST(hashmap, grows_and_preserves_entries) {
    struct PicoHashMap* map = pico_hashmap_init_with_capacity(4);
    int values[10];
    const char* keys[] = {
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    };

    for(int i = 0; i < 10; i++) {
        values[i] = i;
        ASSERT_TRUE(pico_hashmap_insert(map, keys[i], &values[i]));
    }

    ASSERT_TRUE(map->capacity > 4);
    ASSERT_EQ(map->size, (size_t)10);

    for(int i = 0; i < 10; i++) {
        ASSERT_TRUE(*(int*)pico_hashmap_get(map, keys[i]) == i);
    }

    pico_hashmap_free(map);
}

UTEST(hashmap, stores_string_values) {
    struct PicoHashMap* map = pico_hashmap_init();
    char* hello = "hello";
    char* pico = "pico";

    ASSERT_TRUE(pico_hashmap_insert(map, "first", hello));
    ASSERT_TRUE(pico_hashmap_insert(map, "second", pico));

    ASSERT_TRUE(pico_hashmap_get(map, "first") == hello);
    ASSERT_TRUE(pico_hashmap_get(map, "second") == pico);

    pico_hashmap_free(map);
}

UTEST(hashmap, stores_tensor_pointer) {
    struct PicoHashMap* map = pico_hashmap_init();
    struct PicoTensor tensor;

    ASSERT_TRUE(pico_hashmap_insert(map, "weight", &tensor));
    ASSERT_TRUE(pico_hashmap_get(map, "weight") == &tensor);

    pico_hashmap_free(map);
}

UTEST(hashmap, stores_float_pointer) {
    struct PicoHashMap* map = pico_hashmap_init();
    float lr = 0.01f;

    ASSERT_TRUE(pico_hashmap_insert(map, "lr", &lr));
    ASSERT_TRUE(*(float*)pico_hashmap_get(map, "lr") == 0.01f);

    pico_hashmap_free(map);
}

UTEST(hashmap, rejects_invalid_inputs) {
    struct PicoHashMap* map = pico_hashmap_init();
    int value = 3;

    ASSERT_FALSE(pico_hashmap_insert(NULL, "x", &value));
    ASSERT_FALSE(pico_hashmap_insert(map, NULL, &value));
    ASSERT_TRUE(pico_hashmap_get(NULL, "x") == NULL);
    ASSERT_TRUE(pico_hashmap_get(map, NULL) == NULL);

    pico_hashmap_free(map);
}
