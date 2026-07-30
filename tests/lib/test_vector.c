/*
 * Tests for the PicoVec (dynamic array of pointers).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * We only store/compare the POINTERS, so dummy stack ints are fine
 * (the vector never dereferences them).
 */

#include "lib/pico_vector.h"
#include "tensor.h"
#include "utest.h"

// init should set size=0, the requested capacity, and a live buffer
UTEST(vector, init_sets_fields) {
    struct PicoVec v;
    pico_vec_init(&v, 4);
    ASSERT_TRUE(v.data != NULL);
    ASSERT_TRUE(v.size == 0);
    ASSERT_TRUE(v.capacity == 4);
    pico_vec_free(&v);
}

// inserts land in order and bump size
UTEST(vector, insert_stores_in_order) {
    int t0, t1, t2;
    struct PicoVec v;
    pico_vec_init(&v, 4);

    pico_vec_push(&v, &t0);
    pico_vec_push(&v, &t1);
    pico_vec_push(&v, &t2);

    ASSERT_TRUE(v.size == 3);
    ASSERT_TRUE(v.data[0] == &t0);
    ASSERT_TRUE(v.data[1] == &t1);
    ASSERT_TRUE(v.data[2] == &t2);

    pico_vec_free(&v);
}

// inserting past the initial capacity should grow (realloc) and keep every
// element intact, in order — this is the case the topo list actually hits
UTEST(vector, grows_and_preserves_elements) {
    int nodes[5];
    struct PicoVec v;
    pico_vec_init(&v, 2);  // deliberately small

    for(int i = 0; i < 5; i++)
        pico_vec_push(&v, &nodes[i]);

    ASSERT_TRUE(v.size == 5);
    ASSERT_TRUE(v.capacity >= 5);  // 2 -> 4 -> 8 by doubling
    for(int i = 0; i < 5; i++)
        ASSERT_TRUE(v.data[i] == &nodes[i]);  // survived the realloc, in order

    pico_vec_free(&v);
}

// free should release the buffer and zero the bookkeeping
UTEST(vector, free_resets_fields) {
    struct PicoVec v;
    pico_vec_init(&v, 4);
    pico_vec_free(&v);

    ASSERT_TRUE(v.data == NULL);
    ASSERT_TRUE(v.size == 0);
    ASSERT_TRUE(v.capacity == 0);
}

// ============================= search

// search returns the index of a present element
UTEST(vector, search_finds_element) {
    int t0, t1, t2;
    struct PicoVec v;
    pico_vec_init(&v, 4);
    pico_vec_push(&v, &t0);
    pico_vec_push(&v, &t1);
    pico_vec_push(&v, &t2);

    ASSERT_TRUE(pico_vec_find(&v, &t0) == 0);
    ASSERT_TRUE(pico_vec_find(&v, &t1) == 1);
    ASSERT_TRUE(pico_vec_find(&v, &t2) == 2);

    pico_vec_free(&v);
}

// search returns -1 for an absent element (and for an empty vector)
UTEST(vector, search_missing_returns_neg1) {
    int t0, other;
    struct PicoVec v;
    pico_vec_init(&v, 4);

    ASSERT_TRUE(pico_vec_find(&v, &t0) == -1);  // empty
    pico_vec_push(&v, &t0);
    ASSERT_TRUE(pico_vec_find(&v, &other) == -1);  // not present

    pico_vec_free(&v);
}

// ============================= reverse

// reversing an empty or single-element vector is a safe no-op
UTEST(vector, reverse_empty_and_single) {
    int t0;
    struct PicoVec v;

    pico_vec_init(&v, 4);
    pico_vec_reverse(&v);  // empty -> no crash
    ASSERT_TRUE(v.size == 0);

    pico_vec_push(&v, &t0);
    pico_vec_reverse(&v);  // single -> unchanged
    ASSERT_TRUE(v.size == 1);
    ASSERT_TRUE(v.data[0] == &t0);

    pico_vec_free(&v);
}

// reverse flips the order (even count)
UTEST(vector, reverse_even) {
    int a, b, c, d;
    struct PicoVec v;
    pico_vec_init(&v, 4);
    pico_vec_push(&v, &a);
    pico_vec_push(&v, &b);
    pico_vec_push(&v, &c);
    pico_vec_push(&v, &d);

    pico_vec_reverse(&v);

    ASSERT_TRUE(v.data[0] == &d);
    ASSERT_TRUE(v.data[1] == &c);
    ASSERT_TRUE(v.data[2] == &b);
    ASSERT_TRUE(v.data[3] == &a);

    pico_vec_free(&v);
}

// reverse flips the order (odd count -> middle stays put)
UTEST(vector, reverse_odd) {
    int a, b, c;
    struct PicoVec v;
    pico_vec_init(&v, 4);
    pico_vec_push(&v, &a);
    pico_vec_push(&v, &b);
    pico_vec_push(&v, &c);

    pico_vec_reverse(&v);

    ASSERT_TRUE(v.data[0] == &c);
    ASSERT_TRUE(v.data[1] == &b);  // middle unchanged
    ASSERT_TRUE(v.data[2] == &a);

    pico_vec_free(&v);
}

// reversing twice gets you back to the original order
UTEST(vector, reverse_twice_is_identity) {
    int a, b, c;
    struct PicoVec v;
    pico_vec_init(&v, 4);
    pico_vec_push(&v, &a);
    pico_vec_push(&v, &b);
    pico_vec_push(&v, &c);

    pico_vec_reverse(&v);
    pico_vec_reverse(&v);

    ASSERT_TRUE(v.data[0] == &a);
    ASSERT_TRUE(v.data[1] == &b);
    ASSERT_TRUE(v.data[2] == &c);

    pico_vec_free(&v);
}

// generic pointer storage: strings stay as char* pointers.
UTEST(vector, stores_strings) {
    struct PicoVec v;
    pico_vec_init(&v, 2);

    char* hello = "hello";
    char* pico = "pico";
    char* data = "data";

    pico_vec_push(&v, hello);
    pico_vec_push(&v, pico);
    pico_vec_push(&v, data);

    ASSERT_TRUE(v.size == 3);
    ASSERT_TRUE(v.data[0] == hello);
    ASSERT_TRUE(v.data[1] == pico);
    ASSERT_TRUE(v.data[2] == data);
    ASSERT_TRUE(pico_vec_find(&v, pico) == 1);

    pico_vec_free(&v);
}

// generic pointer storage: tensors are just another pointer type to PicoVec.
UTEST(vector, stores_tensor_pointers) {
    struct PicoTensor t0, t1;
    struct PicoVec v;
    pico_vec_init(&v, 2);

    pico_vec_push(&v, &t0);
    pico_vec_push(&v, &t1);

    ASSERT_TRUE(v.data[0] == &t0);
    ASSERT_TRUE(v.data[1] == &t1);
    ASSERT_TRUE(pico_vec_find(&v, &t1) == 1);

    pico_vec_free(&v);
}

// generic pointer storage: floats work when we store their addresses.
UTEST(vector, stores_float_pointers) {
    float a = 1.5f;
    float b = 2.5f;
    float c = 3.5f;
    struct PicoVec v;
    pico_vec_init(&v, 2);

    pico_vec_push(&v, &a);
    pico_vec_push(&v, &b);
    pico_vec_push(&v, &c);

    ASSERT_TRUE(*(float*)v.data[0] == 1.5f);
    ASSERT_TRUE(*(float*)v.data[1] == 2.5f);
    ASSERT_TRUE(*(float*)v.data[2] == 3.5f);

    pico_vec_free(&v);
}

// mixed pointers are allowed because PicoVec does not inspect or own elements.
UTEST(vector, stores_mixed_pointer_types) {
    int count = 7;
    float lr = 0.01f;
    char* name = "sgd";
    struct PicoTensor tensor;
    struct PicoVec v;
    pico_vec_init(&v, 2);

    pico_vec_push(&v, &count);
    pico_vec_push(&v, &lr);
    pico_vec_push(&v, name);
    pico_vec_push(&v, &tensor);

    ASSERT_TRUE(*(int*)v.data[0] == 7);
    ASSERT_TRUE(*(float*)v.data[1] == 0.01f);
    ASSERT_TRUE(v.data[2] == name);
    ASSERT_TRUE(v.data[3] == &tensor);

    pico_vec_free(&v);
}
