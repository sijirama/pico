/*
 * Tests for the PicoLinear layer (pico_nn_linear_init / _forward / _free).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 *
 * These describe what a Linear layer SHOULD do. Some FAIL on purpose against the
 * current code — they're the punch-list. Each failing one says WHY in a comment.
 * Convention (matches the working forward: input @ weights, no transpose):
 *   input   : [batch, in_features]
 *   weights : [in_features, out_features]
 *   bias    : [out_features]  (one value per output neuron, broadcast over batch)
 *   output  : [batch, out_features]
 *
 * Each test calls pico_nn_linear_free(fc) — run `make asan` to see whether the
 * layer's params are actually released (they leak while _free is a no-op).
 */
#include <stdbool.h>

#include "arena.h"
#include "nn/linear.h"
#include "tensor.h"
#include "utest.h"

// ---- anchors: the parts that already work --------------------------------

// init records the feature dims it was given
UTEST(linear, init_sets_dims) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, false);
    ASSERT_TRUE(fc != NULL);
    ASSERT_EQ(fc->in_features, 3);
    ASSERT_EQ(fc->out_features, 4);

    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// bias=false must leave bias NULL (forward then skips the add)
UTEST(linear, no_bias_is_null) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, false);
    ASSERT_TRUE(fc->bias == NULL);

    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// weights must be shaped so `input[.,in] @ weights` works: weights->shape[0]==in,
// weights->shape[1]==out, numel == in*out.  (matmul checks input_last == W->shape[0])
UTEST(linear, weights_shape) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, false);
    ASSERT_TRUE(fc->weights != NULL);
    ASSERT_EQ(fc->weights->ndim, 2);
    ASSERT_TRUE(fc->weights->shape[0] == 3);  // in_features (the matmul contraction dim)
    ASSERT_TRUE(fc->weights->shape[1] == 4);  // out_features
    ASSERT_EQ(fc->weights->numel, 12);

    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// forward, no bias: [2,3] @ layer(3->4) -> [2,4]
UTEST(linear, forward_shape_no_bias) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, false);

    int64_t si[] = {2, 3};
    struct PicoTensor* x = pico_param(si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 4);
    ASSERT_EQ(out->numel, 8);

    pico_free(x);
    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// forward math, no bias: identity weights pass the input straight through.
// x=[3,5] (1x2), W=I (2x2) -> out=[3,5]
UTEST(linear, forward_values_identity) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(2, 2, false);
    fc->weights->data[0] = 1.0f;  // [[1,0],
    fc->weights->data[1] = 0.0f;  //  [0,1]]
    fc->weights->data[2] = 0.0f;
    fc->weights->data[3] = 1.0f;

    int64_t si[] = {1, 2};
    struct PicoTensor* x = pico_param(si, 2);
    x->data[0] = 3.0f;
    x->data[1] = 5.0f;

    struct PicoTensor* out = pico_nn_linear_forward(fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out->data[0] == 3.0f);
    ASSERT_TRUE(out->data[1] == 5.0f);

    pico_free(x);
    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// a mismatched input (last dim != in_features) must be rejected, returning NULL
// (not crash). [2,5] into a layer expecting in_features=3.
UTEST(linear, forward_incompatible_returns_null) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, false);

    int64_t si[] = {2, 5};
    struct PicoTensor* x = pico_param(si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(fc, x);
    ASSERT_TRUE(out == NULL);

    pico_free(x);
    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// params must be PERSISTENT (pico_param) so a per-step arena reset can't wipe them
// and the optimizer can hold a stable pointer. (fixed: init now uses pico_param.)
UTEST(linear, weights_are_persistent) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, true);
    ASSERT_TRUE(fc->weights->is_persistent == 1);
    ASSERT_TRUE(fc->bias->is_persistent == 1);

    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// bias is ONE value per output neuron -> numel == out_features (4).
// (fixed: init now builds bias [out_features, 1] = numel 4, not the weights shape.)
UTEST(linear, bias_is_per_output_feature) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, true);
    ASSERT_TRUE(fc->bias != NULL);
    ASSERT_EQ(fc->bias->numel, 4);  // out_features

    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}

// FAILS: forward WITH bias must give [2,4] for a [2,3] input. bias is currently
// [out,1] = [4,1], so the add is (2,4)+(4,1): last dims 4 vs 1 broadcast fine, but
// dim0 is 2 vs 4 -> NOT broadcastable -> pico_add returns NULL -> forward NULL.
// the out_features axis is on the WRONG side. bias should be [out_features] (pads to
// [1, out_features]) or [1, out_features], so it stretches DOWN the batch dim.
UTEST(linear, forward_with_bias_shape) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    struct PicoLinear* fc = pico_nn_linear_init(3, 4, true);

    int64_t si[] = {2, 3};
    struct PicoTensor* x = pico_param(si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(fc, x);
    ASSERT_TRUE(out != NULL);  // currently NULL -> fails here
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 4);

    pico_free(x);
    pico_nn_linear_free(fc);
    arena_ctx_pop();
    arena_destroy(ar);
}
