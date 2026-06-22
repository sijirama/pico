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

// multi-element backward: grad is sized to numel, so each element gets its own
// gradient slot. (this used to overflow when grad was 1 float - now fixed.)
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

// ============================= sub

// pico_sub should wire the graph the same way add does
UTEST(autograd, sub_wires_graph) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_sub(a, b);

    ASSERT_EQ(c->num_parents, 2);
    ASSERT_TRUE(c->parents[0] == a);
    ASSERT_TRUE(c->parents[1] == b);
    ASSERT_TRUE(c->_backward != NULL);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// d(a-b)/da = +1 and d(a-b)/db = -1: upstream flows straight to a, NEGATED to b
UTEST(autograd, sub_backward_signs) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    a->data[0] = 5.0f;
    struct PicoTensor* b = pico_param(s, 1);
    b->data[0] = 3.0f;

    struct PicoTensor* c = pico_sub(a, b);
    ASSERT_TRUE(c->data[0] == 2.0f);  // forward sanity: 5 - 3

    c->grad[0] = 1.0f;  // upstream gradient
    c->_backward(c);

    ASSERT_TRUE(a->grad[0] == 1.0f);   // +upstream
    ASSERT_TRUE(b->grad[0] == -1.0f);  // -upstream (the sign flip)

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// sub backward should accumulate too: grads keep the sign on each call
UTEST(autograd, sub_backward_accumulates) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_sub(a, b);

    c->grad[0] = 1.0f;
    c->_backward(c);
    c->_backward(c);

    ASSERT_TRUE(a->grad[0] == 2.0f);   // +1 +1
    ASSERT_TRUE(b->grad[0] == -2.0f);  // -1 -1

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// multi-element sub: per-element subtraction + per-element signed gradients
UTEST(autograd, sub_backward_multi_element) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_sub(a, b);

    for(int i = 0; i < 3; i++)
        c->grad[i] = 1.0f;
    c->_backward(c);

    for(int i = 0; i < 3; i++) {
        ASSERT_TRUE(a->grad[i] == 1.0f);
        ASSERT_TRUE(b->grad[i] == -1.0f);
    }

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// ============================= pico_backward (full graph traversal)
// these assume pico_backward seeds entry->grad = 1 itself (micrograd style).

// one op: backward from c flows the seed to both leaves -> a.grad=1, b.grad=1
UTEST(autograd, backward_simple) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    a->data[0] = 2.0f;
    struct PicoTensor* b = pico_param(s, 1);
    b->data[0] = 3.0f;

    struct PicoTensor* c = pico_add(a, b);
    pico_backward(ar, c);

    ASSERT_TRUE(a->grad[0] == 1.0f);
    ASSERT_TRUE(b->grad[0] == 1.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// multi-level chain with a reused leaf:  d = (a + b) + b
//   d.grad=1 -> c.grad=1, b.grad+=1 ; then c -> a.grad=1, b.grad+=1
//   so a.grad=1, b.grad=2 (b feeds two paths), c.grad=1
UTEST(autograd, backward_chain_reuses_leaf) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_add(a, b);
    struct PicoTensor* d = pico_add(c, b);

    pico_backward(ar, d);

    ASSERT_TRUE(c->grad[0] == 1.0f);
    ASSERT_TRUE(a->grad[0] == 1.0f);
    ASSERT_TRUE(b->grad[0] == 2.0f);  // b used in BOTH c and d -> accumulates

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// shared INTERNAL node:  out = c + c  where c = a + b
//   out has c as both parents -> c.grad = 2 ; c -> a.grad=2, b.grad=2
//   if the topo dedup is broken, c's _backward runs twice -> a.grad=4 (wrong)
UTEST(autograd, backward_shared_internal_node) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_add(a, b);
    struct PicoTensor* out = pico_add(c, c);

    pico_backward(ar, out);

    ASSERT_TRUE(a->grad[0] == 2.0f);
    ASSERT_TRUE(b->grad[0] == 2.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// ============================= matmul forward (pico_mul)
// classic (2,3) @ (3,2) -> (2,2):
//   [1 2 3]   [ 7  8]     [ 58  64]
//   [4 5 6] @ [ 9 10]  =  [139 154]
//             [11 12]
// (matmul uses += into out->data, so out->data MUST be zeroed first.)
UTEST(matmul, forward_2x3_times_3x2) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t sa[] = {2, 3};
    struct PicoTensor* a = pico_param(sa, 2);
    float av[] = {1, 2, 3, 4, 5, 6};
    for(int i = 0; i < 6; i++) a->data[i] = av[i];

    int64_t sb[] = {3, 2};
    struct PicoTensor* b = pico_param(sb, 2);
    float bv[] = {7, 8, 9, 10, 11, 12};
    for(int i = 0; i < 6; i++) b->data[i] = bv[i];

    struct PicoTensor* c = pico_mul(a, b);

    ASSERT_TRUE(c->shape[0] == 2);
    ASSERT_TRUE(c->shape[1] == 2);
    ASSERT_TRUE(c->data[0] == 58.0f);
    ASSERT_TRUE(c->data[1] == 64.0f);
    ASSERT_TRUE(c->data[2] == 139.0f);
    ASSERT_TRUE(c->data[3] == 154.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// square matmul too: (2,2) @ (2,2)
//   [1 2]   [5 6]   [19 22]
//   [3 4] @ [7 8] = [43 50]
UTEST(matmul, forward_square) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 2};
    struct PicoTensor* a = pico_param(s, 2);
    float av[] = {1, 2, 3, 4};
    for(int i = 0; i < 4; i++) a->data[i] = av[i];

    struct PicoTensor* b = pico_param(s, 2);
    float bv[] = {5, 6, 7, 8};
    for(int i = 0; i < 4; i++) b->data[i] = bv[i];

    struct PicoTensor* c = pico_mul(a, b);

    ASSERT_TRUE(c->data[0] == 19.0f);
    ASSERT_TRUE(c->data[1] == 22.0f);
    ASSERT_TRUE(c->data[2] == 43.0f);
    ASSERT_TRUE(c->data[3] == 50.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}
