/*
 * Tests for the PicoTensorDVector (dynamic array of PicoTensor*).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * We only store/compare the POINTERS, so dummy stack tensors are fine
 * (the vector never dereferences them).
 */

#include "lib/pico_vector.h"
#include "utest.h"

// init should set size=0, the requested capacity, and a live buffer
UTEST(vector, init_sets_fields) {
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    ASSERT_TRUE(v.data != NULL);
    ASSERT_TRUE(v.size == 0);
    ASSERT_TRUE(v.capacity == 4);
    free_pico_tensor_d_vector(&v);
}

// inserts land in order and bump size
UTEST(vector, insert_stores_in_order) {
    struct PicoTensor t0, t1, t2;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);

    insert_pico_tensor_d_vector(&v, &t0);
    insert_pico_tensor_d_vector(&v, &t1);
    insert_pico_tensor_d_vector(&v, &t2);

    ASSERT_TRUE(v.size == 3);
    ASSERT_TRUE(v.data[0] == &t0);
    ASSERT_TRUE(v.data[1] == &t1);
    ASSERT_TRUE(v.data[2] == &t2);

    free_pico_tensor_d_vector(&v);
}

// inserting past the initial capacity should grow (realloc) and keep every
// element intact, in order — this is the case the topo list actually hits
UTEST(vector, grows_and_preserves_elements) {
    struct PicoTensor nodes[5];
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 2);  // deliberately small

    for(int i = 0; i < 5; i++)
        insert_pico_tensor_d_vector(&v, &nodes[i]);

    ASSERT_TRUE(v.size == 5);
    ASSERT_TRUE(v.capacity >= 5);  // 2 -> 4 -> 8 by doubling
    for(int i = 0; i < 5; i++)
        ASSERT_TRUE(v.data[i] == &nodes[i]);  // survived the realloc, in order

    free_pico_tensor_d_vector(&v);
}

// free should release the buffer and zero the bookkeeping
UTEST(vector, free_resets_fields) {
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    free_pico_tensor_d_vector(&v);

    ASSERT_TRUE(v.data == NULL);
    ASSERT_TRUE(v.size == 0);
    ASSERT_TRUE(v.capacity == 0);
}

// ============================= search

// search returns the index of a present element
UTEST(vector, search_finds_element) {
    struct PicoTensor t0, t1, t2;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    insert_pico_tensor_d_vector(&v, &t0);
    insert_pico_tensor_d_vector(&v, &t1);
    insert_pico_tensor_d_vector(&v, &t2);

    ASSERT_TRUE(search_pico_tensor_d_vector(&v, &t0) == 0);
    ASSERT_TRUE(search_pico_tensor_d_vector(&v, &t1) == 1);
    ASSERT_TRUE(search_pico_tensor_d_vector(&v, &t2) == 2);

    free_pico_tensor_d_vector(&v);
}

// search returns -1 for an absent element (and for an empty vector)
UTEST(vector, search_missing_returns_neg1) {
    struct PicoTensor t0, other;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);

    ASSERT_TRUE(search_pico_tensor_d_vector(&v, &t0) == -1);  // empty
    insert_pico_tensor_d_vector(&v, &t0);
    ASSERT_TRUE(search_pico_tensor_d_vector(&v, &other) == -1);  // not present

    free_pico_tensor_d_vector(&v);
}

// ============================= reverse

// reversing an empty or single-element vector is a safe no-op
UTEST(vector, reverse_empty_and_single) {
    struct PicoTensor t0;
    struct PicoTensorDVector v;

    init_pico_tensor_d_vector(&v, 4);
    reverse_pico_tensor_d_vector(&v);  // empty -> no crash
    ASSERT_TRUE(v.size == 0);

    insert_pico_tensor_d_vector(&v, &t0);
    reverse_pico_tensor_d_vector(&v);  // single -> unchanged
    ASSERT_TRUE(v.size == 1);
    ASSERT_TRUE(v.data[0] == &t0);

    free_pico_tensor_d_vector(&v);
}

// reverse flips the order (even count)
UTEST(vector, reverse_even) {
    struct PicoTensor a, b, c, d;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    insert_pico_tensor_d_vector(&v, &a);
    insert_pico_tensor_d_vector(&v, &b);
    insert_pico_tensor_d_vector(&v, &c);
    insert_pico_tensor_d_vector(&v, &d);

    reverse_pico_tensor_d_vector(&v);

    ASSERT_TRUE(v.data[0] == &d);
    ASSERT_TRUE(v.data[1] == &c);
    ASSERT_TRUE(v.data[2] == &b);
    ASSERT_TRUE(v.data[3] == &a);

    free_pico_tensor_d_vector(&v);
}

// reverse flips the order (odd count -> middle stays put)
UTEST(vector, reverse_odd) {
    struct PicoTensor a, b, c;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    insert_pico_tensor_d_vector(&v, &a);
    insert_pico_tensor_d_vector(&v, &b);
    insert_pico_tensor_d_vector(&v, &c);

    reverse_pico_tensor_d_vector(&v);

    ASSERT_TRUE(v.data[0] == &c);
    ASSERT_TRUE(v.data[1] == &b);  // middle unchanged
    ASSERT_TRUE(v.data[2] == &a);

    free_pico_tensor_d_vector(&v);
}

// reversing twice gets you back to the original order
UTEST(vector, reverse_twice_is_identity) {
    struct PicoTensor a, b, c;
    struct PicoTensorDVector v;
    init_pico_tensor_d_vector(&v, 4);
    insert_pico_tensor_d_vector(&v, &a);
    insert_pico_tensor_d_vector(&v, &b);
    insert_pico_tensor_d_vector(&v, &c);

    reverse_pico_tensor_d_vector(&v);
    reverse_pico_tensor_d_vector(&v);

    ASSERT_TRUE(v.data[0] == &a);
    ASSERT_TRUE(v.data[1] == &b);
    ASSERT_TRUE(v.data[2] == &c);

    free_pico_tensor_d_vector(&v);
}
