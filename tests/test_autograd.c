/*
 * Tests for autograd (pico_add_backward) + the graph wiring in pico_add.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "arena.h"
#include "autograd.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// pico_add should wire the graph: two parents + a backward fn attached
UTEST(autograd, add_wires_graph) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_add(a, b);

    ASSERT_EQ(c->num_parents, 2);
    ASSERT_TRUE(c->parents[0] == a);
    ASSERT_TRUE(c->parents[1] == b);
    ASSERT_TRUE(c->_backward != NULL);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// d(a+b)/da = 1, so the upstream grad flows unchanged into BOTH parents
UTEST(autograd, add_backward_flows_to_parents) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    a->data[0] = 2.0f;
    struct PicoTensor* b = pico_param(s, 1);
    b->data[0] = 3.0f;

    struct PicoTensor* c = pico_add(a, b);
    ASSERT_TRUE(c->data[0] == 5.0f);  // forward sanity

    c->grad[0] = 1.0f;  // upstream gradient
    c->_backward(c);

    ASSERT_TRUE(a->grad[0] == 1.0f);
    ASSERT_TRUE(b->grad[0] == 1.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// calling backward twice must ACCUMULATE (+=), not overwrite
UTEST(autograd, add_backward_accumulates) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_add(a, b);

    c->grad[0] = 1.0f;
    c->_backward(c);
    c->_backward(c);

    ASSERT_TRUE(a->grad[0] == 2.0f);  // 1 + 1
    ASSERT_TRUE(b->grad[0] == 2.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// multi-element backward. needs grad sized to numel, but grad is currently
// allocated as 1 float -> writing a->grad[1..] overflows the param's grad buffer.
// PASSES (maybe flaky) under `make test`; FAILS under `make asan` (heap overflow).
// fix: grad = calloc(numel, sizeof(float)) and arena_alloc(arena, numel*sizeof(float)).
UTEST(autograd, add_backward_multi_element) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_add(a, b);

    for(int i = 0; i < 3; i++)
        c->grad[i] = 1.0f;
    c->_backward(c);

    for(int i = 0; i < 3; i++) {
        ASSERT_TRUE(a->grad[i] == 1.0f);
        ASSERT_TRUE(b->grad[i] == 1.0f);
    }

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}
