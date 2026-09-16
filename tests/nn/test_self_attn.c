#include "utest.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "global.h"
#include "loss/loss.h"
#include "nn/self-attn.h"
#include "optim/optim.h"
#include "tensor.h"

#define ASSERT_NEAR_FLOAT(actual, expected) ASSERT_NEAR((actual), (expected), 1e-5f)

void pico_nn_attn_causal_mask(struct PicoTensor* table);

static void fill_identity_2x2(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = 0.0f;
    }

    tensor->data[0] = 1.0f;
    tensor->data[3] = 1.0f;
}

static void fill_identity(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = 0.0f;
    }

    int64_t rows = tensor->shape[0];
    int64_t cols = tensor->shape[1];
    int64_t diag = rows < cols ? rows : cols;
    for(int64_t i = 0; i < diag; i++) {
        tensor->data[i * tensor->strides[0] + i * tensor->strides[1]] = 1.0f;
    }
}

static int run_attention_3d_forward_child(void) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "child.attn", 2, 1, 2);
    int64_t input_shape[] = {1, 2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    if(attn == NULL) {
        pico_shutdown(ctx);
        return 1;
    }

    fill_identity(attn->Q);
    fill_identity(attn->K);
    fill_identity(attn->V);
    fill_identity(attn->O);

    struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 3, input_values);
    struct PicoTensor* out = pico_nn_attn_forward(ctx, attn, input);
    bool ok = out != NULL && out->ndim == 3 && out->shape[0] == 1 && out->shape[1] == 2 && out->shape[2] == 2;

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
    return ok ? 0 : 1;
}

static int run_attention_multihead_forward_child(void) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "mh.attn", 4, 2, 2);
    int64_t input_shape[] = {2, 4};
    float input_values[] = {
        1.0f, 0.0f, 10.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 20.0f,
    };
    bool ok = false;

    if(attn != NULL) {
        fill_identity(attn->Q);
        fill_identity(attn->K);
        fill_identity(attn->V);
        fill_identity(attn->O);

        struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 2, input_values);
        struct PicoTensor* out = pico_nn_attn_forward(ctx, attn, input);
        ok = out != NULL && out->ndim == 2 && out->shape[0] == 2 && out->shape[1] == 4 &&
             fabsf(out->data[0] - 1.0f) < 1e-5f && fabsf(out->data[1] - 0.0f) < 1e-5f &&
             fabsf(out->data[2] - 10.0f) < 1e-5f && fabsf(out->data[3] - 0.0f) < 1e-5f;
    }

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
    return ok ? 0 : 1;
}

static int run_attention_backward_child(void) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 2, 1, 2);
    int64_t input_shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float target_values[] = {
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    bool has_grad = false;

    if(attn != NULL) {
        fill_identity(attn->Q);
        fill_identity(attn->K);
        fill_identity(attn->V);
        fill_identity(attn->O);

        struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 2, input_values);
        struct PicoTensor* target = pico_tensor_from_data(ctx, input_shape, 2, target_values);
        struct PicoTensor* out = pico_nn_attn_forward(ctx, attn, input);
        struct PicoMSELoss mse = {.reduction = MEAN};
        struct PicoTensor* loss = pico_mse_loss(ctx, &mse, out, target);
        pico_backward(ctx, loss);

        for(int i = 0; i < attn->O->numel; i++) {
            if(fabsf(attn->O->grad[i]) > 1e-6f) {
                has_grad = true;
            }
        }
    }

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
    return has_grad ? 0 : 1;
}

static int run_attention_training_child(void) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 2, 1, 2);
    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    int64_t input_shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float target_values[] = {
        0.0f, 1.0f,
        1.0f, 0.0f,
    };
    bool ok = false;
    float before = 0.0f;
    float after = 0.0f;

    if(attn != NULL && opt != NULL) {
        fill_identity(attn->Q);
        fill_identity(attn->K);
        fill_identity(attn->V);
        fill_identity(attn->O);

        struct PicoMSELoss mse = {.reduction = MEAN};
        struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 2, input_values);
        struct PicoTensor* target = pico_tensor_from_data(ctx, input_shape, 2, target_values);

        struct PicoTensor* pred1 = pico_nn_attn_forward(ctx, attn, input);
        struct PicoTensor* loss1 = pico_mse_loss(ctx, &mse, pred1, target);
        before = loss1->data[0];

        pico_optim_sgd_zero_grad(ctx, opt);
        pico_backward(ctx, loss1);
        pico_optim_sgd_step(ctx, opt);

        struct PicoTensor* pred2 = pico_nn_attn_forward(ctx, attn, input);
        struct PicoTensor* loss2 = pico_mse_loss(ctx, &mse, pred2, target);
        after = loss2->data[0];

        ok = isfinite(before) && isfinite(after) && after < before;
    }

    pico_optim_sgd_free(opt);
    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
    return ok ? 0 : 1;
}

static bool child_exits_successfully(int (*fn)(void)) {
    pid_t pid = fork();
    if(pid == 0) {
        _exit(fn());
    }
    if(pid < 0) {
        return false;
    }

    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

UTEST(self_attn, init_sets_dims_and_projection_shapes) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "block.attn", 12, 3, 4);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(attn->embed_dim, 12);
    ASSERT_EQ(attn->num_of_heads, 3);
    ASSERT_EQ(attn->d_k, 4);

    ASSERT_EQ(attn->Q->shape[0], 12);
    ASSERT_EQ(attn->Q->shape[1], 12);
    ASSERT_EQ(attn->K->shape[0], 12);
    ASSERT_EQ(attn->K->shape[1], 12);
    ASSERT_EQ(attn->V->shape[0], 12);
    ASSERT_EQ(attn->V->shape[1], 12);
    ASSERT_EQ(attn->O->shape[0], 12);
    ASSERT_EQ(attn->O->shape[1], 12);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_params_are_heap_tensors) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "block.attn", 12, 3, 4);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(attn->Q->storage, PICO_TENSOR_STORAGE_HEAP);
    ASSERT_EQ(attn->K->storage, PICO_TENSOR_STORAGE_HEAP);
    ASSERT_EQ(attn->V->storage, PICO_TENSOR_STORAGE_HEAP);
    ASSERT_EQ(attn->O->storage, PICO_TENSOR_STORAGE_HEAP);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_allows_head_projection_dim_different_from_embed_dim) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "wide.attn", 8, 4, 3);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(attn->Q->shape[0], 8);
    ASSERT_EQ(attn->Q->shape[1], 12);
    ASSERT_EQ(attn->K->shape[0], 8);
    ASSERT_EQ(attn->K->shape[1], 12);
    ASSERT_EQ(attn->V->shape[0], 8);
    ASSERT_EQ(attn->V->shape[1], 12);
    ASSERT_EQ(attn->O->shape[0], 12);
    ASSERT_EQ(attn->O->shape[1], 8);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_registers_named_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 4, 2, 2);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(ctx->params.size, (size_t)4);
    ASSERT_STREQ(attn->Q->name, "tiny.attn.q_proj.weight");
    ASSERT_STREQ(attn->K->name, "tiny.attn.k_proj.weight");
    ASSERT_STREQ(attn->V->name, "tiny.attn.v_proj.weight");
    ASSERT_STREQ(attn->O->name, "tiny.attn.out_proj.weight");
    ASSERT_TRUE(ctx->params.data[0] == attn->Q);
    ASSERT_TRUE(ctx->params.data[1] == attn->K);
    ASSERT_TRUE(ctx->params.data[2] == attn->V);
    ASSERT_TRUE(ctx->params.data[3] == attn->O);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_rejects_invalid_inputs) {
    struct PicoContext* ctx = pico_init_verbose(false);

    ASSERT_TRUE(pico_nn_attn_init(NULL, "attn", 4, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, NULL, 4, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 0, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 4, 0, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 4, 2, 0) == NULL);

    ASSERT_EQ(ctx->params.size, (size_t)0);
    pico_shutdown(ctx);
}

UTEST(self_attn, rope_rotates_qk_pairs_by_position) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 4};
    float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
    };

    struct PicoTensor* q = pico_tensor_from_data(ctx, shape, 2, values);
    pico_nn_attn_apply_rope(q, 1, 4);

    ASSERT_NEAR_FLOAT(q->data[0], 1.0f);
    ASSERT_NEAR_FLOAT(q->data[1], 2.0f);
    ASSERT_NEAR_FLOAT(q->data[2], 3.0f);
    ASSERT_NEAR_FLOAT(q->data[3], 4.0f);

    ASSERT_NEAR_FLOAT(q->data[4], cosf(1.0f));
    ASSERT_NEAR_FLOAT(q->data[5], sinf(1.0f));
    ASSERT_NEAR_FLOAT(q->data[6], cosf(0.01f));
    ASSERT_NEAR_FLOAT(q->data[7], sinf(0.01f));

    pico_shutdown(ctx);
}

UTEST(self_attn, forward_smoke_returns_embed_dim_for_square_2d_input) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 2, 1, 2);
    int64_t input_shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    ASSERT_TRUE(attn != NULL);
    fill_identity_2x2(attn->Q);
    fill_identity_2x2(attn->K);
    fill_identity_2x2(attn->V);
    fill_identity_2x2(attn->O);

    struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 2, input_values);
    struct PicoTensor* out = pico_nn_attn_forward(ctx, attn, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->shape[0], 2);
    ASSERT_EQ(out->shape[1], 2);
    for(int i = 0; i < out->numel; i++) {
        ASSERT_TRUE(isfinite(out->data[i]));
    }

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, causal_mask_sets_future_scores_to_negative_infinity) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {1, 1, 3, 3};
    float values[] = {
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f,
    };

    struct PicoTensor* scores = pico_tensor_from_data(ctx, shape, 4, values);
    pico_nn_attn_causal_mask(scores);

    float actual[9];
    for(int i = 0; i < 9; i++) {
        actual[i] = scores->data[i];
    }

    pico_shutdown(ctx);

    ASSERT_NEAR_FLOAT(actual[0], 0.0f);
    ASSERT_TRUE(isinf(actual[1]) && actual[1] < 0.0f);
    ASSERT_TRUE(isinf(actual[2]) && actual[2] < 0.0f);
    ASSERT_NEAR_FLOAT(actual[3], 3.0f);
    ASSERT_NEAR_FLOAT(actual[4], 4.0f);
    ASSERT_TRUE(isinf(actual[5]) && actual[5] < 0.0f);
    ASSERT_NEAR_FLOAT(actual[6], 6.0f);
    ASSERT_NEAR_FLOAT(actual[7], 7.0f);
    ASSERT_NEAR_FLOAT(actual[8], 8.0f);
}

UTEST(self_attn, forward_matches_hand_computed_single_head_causal_attention) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 2, 1, 2);
    int64_t input_shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float actual[4] = {0};
    bool ok = false;

    if(attn != NULL) {
        fill_identity(attn->Q);
        fill_identity(attn->K);
        fill_identity(attn->V);
        fill_identity(attn->O);

        struct PicoTensor* input = pico_tensor_from_data(ctx, input_shape, 2, input_values);
        struct PicoTensor* out = pico_nn_attn_forward(ctx, attn, input);
        ok = out != NULL && out->ndim == 2 && out->shape[0] == 2 && out->shape[1] == 2;
        if(ok) {
            for(int i = 0; i < 4; i++) {
                actual[i] = out->data[i];
            }
        }
    }

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);

    float score_10 = -sinf(1.0f) / sqrtf(2.0f);
    float score_11 = 1.0f / sqrtf(2.0f);
    float denom = expf(score_10) + expf(score_11);
    float w10 = expf(score_10) / denom;
    float w11 = expf(score_11) / denom;

    ASSERT_TRUE(ok);
    ASSERT_NEAR_FLOAT(actual[0], 1.0f);
    ASSERT_NEAR_FLOAT(actual[1], 0.0f);
    ASSERT_NEAR_FLOAT(actual[2], w10);
    ASSERT_NEAR_FLOAT(actual[3], w11);
}

UTEST(self_attn, forward_accepts_real_batch_sequence_embed_input) {
    ASSERT_TRUE(child_exits_successfully(run_attention_3d_forward_child));
}

UTEST(self_attn, forward_keeps_multihead_channels_separate) {
    ASSERT_TRUE(child_exits_successfully(run_attention_multihead_forward_child));
}

UTEST(self_attn, backward_populates_projection_grads) {
    ASSERT_TRUE(child_exits_successfully(run_attention_backward_child));
}

UTEST(self_attn, one_sgd_step_through_attention_lowers_loss) {
    ASSERT_TRUE(child_exits_successfully(run_attention_training_child));
}
