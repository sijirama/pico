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

// ============================= matmul forward (pico_matmul)
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

    struct PicoTensor* c = pico_matmul(a, b);

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

    struct PicoTensor* c = pico_matmul(a, b);

    ASSERT_TRUE(c->data[0] == 19.0f);
    ASSERT_TRUE(c->data[1] == 22.0f);
    ASSERT_TRUE(c->data[2] == 43.0f);
    ASSERT_TRUE(c->data[3] == 50.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// matmul backward: C = A·B,  dA = dC·Bᵀ,  dB = Aᵀ·dC, with all-ones upstream.
//   A=[[1,2],[3,4]]  B=[[5,6],[7,8]]   seed dC = ones(2,2)
//   dA = ones·Bᵀ = [[11,15],[11,15]]   (row sums of B: 5+6, 7+8)
//   dB = Aᵀ·ones = [[4,4],[6,6]]       (col sums of A: 1+3, 2+4)
UTEST(matmul, backward_square) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 2};
    struct PicoTensor* a = pico_param(s, 2);
    float av[] = {1, 2, 3, 4};
    for(int i = 0; i < 4; i++) a->data[i] = av[i];

    struct PicoTensor* b = pico_param(s, 2);
    float bv[] = {5, 6, 7, 8};
    for(int i = 0; i < 4; i++) b->data[i] = bv[i];

    struct PicoTensor* c = pico_matmul(a, b);
    pico_backward(ar, c);  // seeds c->grad = 1

    ASSERT_TRUE(a->grad[0] == 11.0f);
    ASSERT_TRUE(a->grad[1] == 15.0f);
    ASSERT_TRUE(a->grad[2] == 11.0f);
    ASSERT_TRUE(a->grad[3] == 15.0f);

    ASSERT_TRUE(b->grad[0] == 4.0f);
    ASSERT_TRUE(b->grad[1] == 4.0f);
    ASSERT_TRUE(b->grad[2] == 6.0f);
    ASSERT_TRUE(b->grad[3] == 6.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// non-square matmul backward: (2,3)@(3,2). this is the case the OLD element-wise
// backward got OUT OF BOUNDS on — so it also confirms the shapes are handled right.
UTEST(matmul, backward_non_square) {
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

    struct PicoTensor* c = pico_matmul(a, b);  // (2,2)
    pico_backward(ar, c);                       // seed dC = ones(2,2)

    // dA = dC·Bᵀ, dC=ones(2,2): each dA[i][k] = sum of row k of B = (b[k][0]+b[k][1])
    //   row0:7+8=15, row1:9+10=19, row2:11+12=23  -> every A row = [15,19,23]
    ASSERT_TRUE(a->grad[0] == 15.0f);
    ASSERT_TRUE(a->grad[1] == 19.0f);
    ASSERT_TRUE(a->grad[2] == 23.0f);
    ASSERT_TRUE(a->grad[3] == 15.0f);
    ASSERT_TRUE(a->grad[4] == 19.0f);
    ASSERT_TRUE(a->grad[5] == 23.0f);

    // dB = Aᵀ·dC: each dB[k][j] = sum of col k of A = (a[0][k]+a[1][k])
    //   col0:1+4=5, col1:2+5=7, col2:3+6=9  -> dB rows = [5,5],[7,7],[9,9]
    ASSERT_TRUE(b->grad[0] == 5.0f);
    ASSERT_TRUE(b->grad[1] == 5.0f);
    ASSERT_TRUE(b->grad[2] == 7.0f);
    ASSERT_TRUE(b->grad[3] == 7.0f);
    ASSERT_TRUE(b->grad[4] == 9.0f);
    ASSERT_TRUE(b->grad[5] == 9.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// ============================= full-backward coverage (via pico_backward)

// deep add chain: L = ((a + b) + c) + d  -> every leaf gets grad 1
UTEST(backward_full, add_deep_chain) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    struct PicoTensor* c = pico_param(s, 1);
    struct PicoTensor* d = pico_param(s, 1);

    struct PicoTensor* L = pico_add(pico_add(pico_add(a, b), c), d);
    pico_backward(ar, L);

    ASSERT_TRUE(a->grad[0] == 1.0f);
    ASSERT_TRUE(b->grad[0] == 1.0f);
    ASSERT_TRUE(c->grad[0] == 1.0f);
    ASSERT_TRUE(d->grad[0] == 1.0f);

    pico_free(a);
    pico_free(b);
    pico_free(c);
    pico_free(d);
    arena_ctx_pop();
    arena_destroy(ar);
}

// diamond: a feeds TWO branches that merge.  p = a+x ; q = a+y ; L = p+q
//   a reaches L through both p and q -> a.grad = 2 ; x.grad = 1 ; y.grad = 1
UTEST(backward_full, add_diamond_reuses_leaf) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* x = pico_param(s, 1);
    struct PicoTensor* y = pico_param(s, 1);

    struct PicoTensor* p = pico_add(a, x);
    struct PicoTensor* q = pico_add(a, y);
    struct PicoTensor* L = pico_add(p, q);
    pico_backward(ar, L);

    ASSERT_TRUE(a->grad[0] == 2.0f);  // two paths to L
    ASSERT_TRUE(x->grad[0] == 1.0f);
    ASSERT_TRUE(y->grad[0] == 1.0f);

    pico_free(a);
    pico_free(x);
    pico_free(y);
    arena_ctx_pop();
    arena_destroy(ar);
}

// matmul shaped as a dot product: (1,3)@(3,1) -> (1,1).  seed dC=1.
//   dA = dC·Bᵀ = B as a row -> a.grad = [4,5,6]
//   dB = Aᵀ·dC = A as a col -> b.grad = [1,2,3]
UTEST(backward_full, matmul_dot_shape) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t sa[] = {1, 3};
    struct PicoTensor* a = pico_param(sa, 2);
    float av[] = {1, 2, 3};
    for(int i = 0; i < 3; i++) a->data[i] = av[i];

    int64_t sb[] = {3, 1};
    struct PicoTensor* b = pico_param(sb, 2);
    float bv[] = {4, 5, 6};
    for(int i = 0; i < 3; i++) b->data[i] = bv[i];

    struct PicoTensor* c = pico_matmul(a, b);
    ASSERT_TRUE(c->data[0] == 32.0f);  // 1*4 + 2*5 + 3*6
    pico_backward(ar, c);

    ASSERT_TRUE(a->grad[0] == 4.0f);
    ASSERT_TRUE(a->grad[1] == 5.0f);
    ASSERT_TRUE(a->grad[2] == 6.0f);
    ASSERT_TRUE(b->grad[0] == 1.0f);
    ASSERT_TRUE(b->grad[1] == 2.0f);
    ASSERT_TRUE(b->grad[2] == 3.0f);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
}

// matmul feeding an add (a "layer" without bias-broadcast): E = (A@B) + C
//   E.grad = ones -> add sends ones to both (A@B) and C
//   then matmul backward with upstream=ones: dA=[11,15,11,15], dB=[4,4,6,6]
//   C is a direct leaf of the add -> C.grad = ones
UTEST(backward_full, matmul_then_add) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 2};
    struct PicoTensor* a = pico_param(s, 2);
    float av[] = {1, 2, 3, 4};
    for(int i = 0; i < 4; i++) a->data[i] = av[i];

    struct PicoTensor* b = pico_param(s, 2);
    float bv[] = {5, 6, 7, 8};
    for(int i = 0; i < 4; i++) b->data[i] = bv[i];

    struct PicoTensor* c = pico_param(s, 2);  // same shape as A@B -> no broadcast

    struct PicoTensor* e = pico_add(pico_matmul(a, b), c);
    pico_backward(ar, e);

    ASSERT_TRUE(a->grad[0] == 11.0f);
    ASSERT_TRUE(a->grad[1] == 15.0f);
    ASSERT_TRUE(a->grad[2] == 11.0f);
    ASSERT_TRUE(a->grad[3] == 15.0f);

    ASSERT_TRUE(b->grad[0] == 4.0f);
    ASSERT_TRUE(b->grad[1] == 4.0f);
    ASSERT_TRUE(b->grad[2] == 6.0f);
    ASSERT_TRUE(b->grad[3] == 6.0f);

    for(int i = 0; i < 4; i++)
        ASSERT_TRUE(c->grad[i] == 1.0f);  // C is added straight in

    pico_free(a);
    pico_free(b);
    pico_free(c);
    arena_ctx_pop();
    arena_destroy(ar);
}

// shared parent through matmul:  C = A @ A  (both parents are A)
//   A=[[1,2],[3,4]], seed dC=ones.  A.grad = dA + dB (same tensor accumulates both):
//   dA = dC·Aᵀ = [[3,7],[3,7]] ,  dB = Aᵀ·dC = [[4,4],[6,6]]
//   sum -> A.grad = [[7,11],[9,13]]
UTEST(backward_full, matmul_shared_parent) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 2};
    struct PicoTensor* a = pico_param(s, 2);
    float av[] = {1, 2, 3, 4};
    for(int i = 0; i < 4; i++) a->data[i] = av[i];

    struct PicoTensor* c = pico_matmul(a, a);
    pico_backward(ar, c);

    ASSERT_TRUE(a->grad[0] == 7.0f);
    ASSERT_TRUE(a->grad[1] == 11.0f);
    ASSERT_TRUE(a->grad[2] == 9.0f);
    ASSERT_TRUE(a->grad[3] == 13.0f);

    pico_free(a);
    arena_ctx_pop();
    arena_destroy(ar);
}
