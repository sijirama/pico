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
 * PicoContext owns layer params. pico_nn_linear_free only frees the layer shell;
 * ctx_destroy frees weights and bias.
 */
#include <stdbool.h>

#include "ctx.h"
#include "global.h"
#include "loss/loss.h"
#include "nn/linear.h"
#include "optim/optim.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// ---- anchors: the parts that already work --------------------------------

// init records the feature dims it was given
UTEST(linear, init_sets_dims) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, false);
    ASSERT_TRUE(fc != NULL);
    ASSERT_EQ(fc->in_features, 3);
    ASSERT_EQ(fc->out_features, 4);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// bias=false must leave bias NULL (forward then skips the add)
UTEST(linear, no_bias_is_null) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, false);
    ASSERT_TRUE(fc->bias == NULL);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// weights must be shaped so `input[.,in] @ weights` works: weights->shape[0]==in,
// weights->shape[1]==out, numel == in*out.  (matmul checks input_last == W->shape[0])
UTEST(linear, weights_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, false);
    ASSERT_TRUE(fc->weights != NULL);
    ASSERT_EQ(fc->weights->ndim, 2);
    ASSERT_TRUE(fc->weights->shape[0] == 3);  // in_features (the matmul contraction dim)
    ASSERT_TRUE(fc->weights->shape[1] == 4);  // out_features
    ASSERT_EQ(fc->weights->numel, 12);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// forward, no bias: [2,3] @ layer(3->4) -> [2,4]
UTEST(linear, forward_shape_no_bias) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, false);

    int64_t si[] = {2, 3};
    struct PicoTensor* x = pico_param(ctx, si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 4);
    ASSERT_EQ(out->numel, 8);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// forward math, no bias: identity weights pass the input straight through.
// x=[3,5] (1x2), W=I (2x2) -> out=[3,5]
UTEST(linear, forward_values_identity) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 2, false);
    fc->weights->data[0] = 1.0f;  // [[1,0],
    fc->weights->data[1] = 0.0f;  //  [0,1]]
    fc->weights->data[2] = 0.0f;
    fc->weights->data[3] = 1.0f;

    int64_t si[] = {1, 2};
    struct PicoTensor* x = pico_param(ctx, si, 2);
    x->data[0] = 3.0f;
    x->data[1] = 5.0f;

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out->data[0] == 3.0f);
    ASSERT_TRUE(out->data[1] == 5.0f);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// a mismatched input (last dim != in_features) must be rejected, returning NULL
// (not crash). [2,5] into a layer expecting in_features=3.
UTEST(linear, forward_incompatible_returns_null) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, false);

    int64_t si[] = {2, 5};
    struct PicoTensor* x = pico_param(ctx, si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out == NULL);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// params must be heap-backed so a per-step arena reset can't wipe them and the
// optimizer can hold a stable pointer.
UTEST(linear, weights_use_heap_storage) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, true);
    ASSERT_EQ(fc->weights->storage, PICO_TENSOR_STORAGE_HEAP);
    ASSERT_EQ(fc->bias->storage, PICO_TENSOR_STORAGE_HEAP);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

UTEST(linear, free_keeps_params_registered_for_context_cleanup) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, true);
    ASSERT_EQ(ctx->params.size, (size_t)2);

    pico_nn_linear_free(fc);

    ASSERT_EQ(ctx->params.size, (size_t)2);
    pico_shutdown(ctx);
}

// bias is ONE value per output neuron -> numel == out_features (4).
// (fixed: init now builds bias [out_features, 1] = numel 4, not the weights shape.)
UTEST(linear, bias_is_per_output_feature) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, true);
    ASSERT_TRUE(fc->bias != NULL);
    ASSERT_EQ(fc->bias->numel, 4);  // out_features

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// FAILS: forward WITH bias must give [2,4] for a [2,3] input. bias is currently
// [out,1] = [4,1], so the add is (2,4)+(4,1): last dims 4 vs 1 broadcast fine, but
// dim0 is 2 vs 4 -> NOT broadcastable -> pico_add returns NULL -> forward NULL.
// the out_features axis is on the WRONG side. bias should be [out_features] (pads to
// [1, out_features]) or [1, out_features], so it stretches DOWN the batch dim.
UTEST(linear, forward_with_bias_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 3, 4, true);

    int64_t si[] = {2, 3};
    struct PicoTensor* x = pico_param(ctx, si, 2);

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out != NULL);  // bias is [out] now -> broadcasts down the batch
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 4);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// ---- forward math WITH bias -----------------------------------------------

// exact value: out = x @ W + b.  W=[3,4] (2x1), b=[10], x=[1,2] -> 1*3+2*4+10 = 21
UTEST(linear, forward_with_bias_values) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 1, true);
    fc->weights->data[0] = 3.0f;
    fc->weights->data[1] = 4.0f;
    fc->bias->data[0] = 10.0f;

    int64_t si[] = {1, 2};
    struct PicoTensor* x = pico_param(ctx, si, 2);
    x->data[0] = 1.0f;
    x->data[1] = 2.0f;

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out->data[0] == 21.0f);  // 3 + 8 + 10

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// the SAME bias is added to every row of a batch (broadcast down dim 0).
// W=[1,1] (2x1), b=[5], x=[[1,1],[2,2]] (2x2) -> [[2+5],[4+5]] = [[7],[9]]
UTEST(linear, bias_broadcasts_over_batch) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 1, true);
    fc->weights->data[0] = 1.0f;
    fc->weights->data[1] = 1.0f;
    fc->bias->data[0] = 5.0f;

    int64_t si[] = {2, 2};
    struct PicoTensor* x = pico_param(ctx, si, 2);
    x->data[0] = 1.0f;
    x->data[1] = 1.0f;
    x->data[2] = 2.0f;
    x->data[3] = 2.0f;

    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->data[0] == 7.0f);  // row 0: 2 + 5
    ASSERT_TRUE(out->data[1] == 9.0f);  // row 1: 4 + 5

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// ---- backward through the layer -------------------------------------------

// after loss.backward(), BOTH weights and bias must carry gradient.
// W=[0,0], b=[0], x=[1,2], target=[5], MSE mean (N=1) -> out=0, dL/dout = 2*(0-5) = -10
//   W.grad = x^T * dout = [-10, -20] ;  bias.grad = sum over batch = -10
UTEST(linear, backward_populates_grads) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 1, true);  // W,b start at 0 (calloc)

    int64_t si[] = {1, 2};
    struct PicoTensor* x = pico_param(ctx, si, 2);
    x->data[0] = 1.0f;
    x->data[1] = 2.0f;

    int64_t st[] = {1, 1};
    struct PicoTensor* target = pico_param(ctx, st, 2);
    target->data[0] = 5.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
    struct PicoTensor* loss = pico_mse_loss(ctx, &mse, out, target);
    pico_backward(ctx, loss);

    ASSERT_TRUE(fc->weights->grad[0] == -10.0f);
    ASSERT_TRUE(fc->weights->grad[1] == -20.0f);
    ASSERT_TRUE(fc->bias->grad[0] == -10.0f);

    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// ---- it TRAINS: full stack (linear -> mse -> backward -> sgd) --------------

// one SGD step must lower the loss. W,b=0, x=[1,2], target=5, lr=0.01:
//   out=0, loss=25 ; after one step out moves toward 5 -> loss < 25
UTEST(linear, trains_one_step_lowers_loss) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 1, true);

    int64_t si[] = {1, 2};
    float x_data[] = {1.0f, 2.0f};
    struct PicoTensor* x = pico_tensor_from_data(ctx, si, 2, x_data);

    int64_t st[] = {1, 1};
    float target_data[] = {5.0f};
    struct PicoTensor* target = pico_tensor_from_data(ctx, st, 2, target_data);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    struct PicoMSELoss mse = {.reduction = MEAN};

    struct PicoTensor* out1 = pico_nn_linear_forward(ctx, fc, x);
    struct PicoTensor* loss1 = pico_mse_loss(ctx, &mse, out1, target);
    float l1 = loss1->data[0];

    pico_optim_sgd_zero_grad(ctx, opt);
    pico_backward(ctx, loss1);
    pico_optim_sgd_step(ctx, opt);

    struct PicoTensor* out2 = pico_nn_linear_forward(ctx, fc, x);
    struct PicoTensor* loss2 = pico_mse_loss(ctx, &mse, out2, target);
    float l2 = loss2->data[0];

    ASSERT_TRUE(l1 == 25.0f);  // sanity: starts where we expect
    ASSERT_TRUE(l2 < l1);      // it learned

    pico_optim_sgd_free(opt);
    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}

// over several steps the loss strictly decreases and converges toward 0 —
// the layer actually fits x=[1,2] -> target=5 using its weights AND bias.
UTEST(linear, trains_converges) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLinear* fc = pico_nn_linear_init(ctx, "fc", 2, 1, true);

    int64_t si[] = {1, 2};
    float x_data[] = {1.0f, 2.0f};
    struct PicoTensor* x = pico_tensor_from_data(ctx, si, 2, x_data);

    int64_t st[] = {1, 1};
    float target_data[] = {5.0f};
    struct PicoTensor* target = pico_tensor_from_data(ctx, st, 2, target_data);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    struct PicoMSELoss mse = {.reduction = MEAN};

    float prev = 1e30f;
    for(int step = 0; step < 25; step++) {
        struct PicoTensor* out = pico_nn_linear_forward(ctx, fc, x);
        struct PicoTensor* loss = pico_mse_loss(ctx, &mse, out, target);
        float l = loss->data[0];
        ASSERT_TRUE(l < prev);  // strictly decreasing every step
        prev = l;

        pico_optim_sgd_zero_grad(ctx, opt);
        pico_backward(ctx, loss);
        pico_optim_sgd_step(ctx, opt);
    }

    ASSERT_TRUE(prev < 1.0f);  // converged close to the target (started at 25)

    pico_optim_sgd_free(opt);
    pico_nn_linear_free(fc);
    pico_shutdown(ctx);
}
