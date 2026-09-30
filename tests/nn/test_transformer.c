#include "utest.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "global.h"
#include "act/activations.h"
#include "loss/loss.h"
#include "nn/attn.h"
#include "nn/embedding.h"
#include "nn/linear.h"
#include "nn/transformer.h"
#include "norm/norm.h"
#include "optim/optim.h"
#include "tensor.h"

#define ASSERT_NEAR_FLOAT(actual, expected) ASSERT_NEAR((actual), (expected), 1e-5f)

static void fill_zero(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = 0.0f;
    }
}

static void fill_identity(struct PicoTensor* tensor) {
    fill_zero(tensor);

    int64_t rows = tensor->shape[0];
    int64_t cols = tensor->shape[1];
    int64_t diag = rows < cols ? rows : cols;
    for(int64_t i = 0; i < diag; i++) {
        tensor->data[i * tensor->strides[0] + i * tensor->strides[1]] = 1.0f;
    }
}

static bool has_param_named(struct PicoContext* ctx, const char* name) {
    for(int i = 0; i < ctx->params.size; i++) {
        struct PicoTensor* param = ctx->params.data[i];
        if(param != NULL && param->name != NULL && strcmp(param->name, name) == 0) {
            return true;
        }
    }

    return false;
}

struct TinyCpuDecoderSmoke {
    struct PicoEmbedding* tok_emb;
    struct PicoRMSNorm* norm_local_0;
    struct PicoAttn* local_0;
    struct PicoRMSNorm* norm_local_1;
    struct PicoAttn* local_1;
    struct PicoRMSNorm* norm_global;
    struct PicoAttn* global;
    struct PicoRMSNorm* norm_ffn;
    struct PicoLinear* ffn_up;
    struct PicoLinear* ffn_gate;
    struct PicoLinear* ffn_down;
    struct PicoLinear* lm_head;
};

static void seed_param_values(struct PicoContext* ctx) {
    for(int p = 0; p < ctx->params.size; p++) {
        struct PicoTensor* tensor = ctx->params.data[p];
        if(tensor == NULL || tensor->data == NULL) {
            continue;
        }

        bool is_norm_weight = tensor->name != NULL && strstr(tensor->name, "norm") != NULL;
        bool is_bias = tensor->name != NULL && strstr(tensor->name, ".bias") != NULL;

        for(int i = 0; i < tensor->numel; i++) {
            if(is_norm_weight) {
                tensor->data[i] = 1.0f;
            } else if(is_bias) {
                tensor->data[i] = 0.0f;
            } else {
                int bucket = ((p + 1) * (i + 3)) % 11;
                tensor->data[i] = ((float)bucket - 5.0f) * 0.02f;
            }
        }
    }
}

static struct TinyCpuDecoderSmoke tiny_cpu_decoder_smoke_init(struct PicoContext* ctx) {
    struct TinyCpuDecoderSmoke model = {0};

    model.tok_emb = pico_embedding_init(ctx, 8, 4);
    model.norm_local_0 = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.local.0.norm", 4, 1e-5f);
    model.local_0 = pico_nn_swa_attn_init(ctx, "tiny.blocks.0.local.0.attn", 4, 2, 2, 2);
    model.norm_local_1 = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.local.1.norm", 4, 1e-5f);
    model.local_1 = pico_nn_swa_attn_init(ctx, "tiny.blocks.0.local.1.attn", 4, 2, 2, 2);
    model.norm_global = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.global.norm", 4, 1e-5f);
    model.global = pico_nn_attn_init(ctx, "tiny.blocks.0.global.attn", 4, 2, 2);
    model.norm_ffn = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.ffn.norm", 4, 1e-5f);
    model.ffn_up = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.up", 4, 12, true);
    model.ffn_gate = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.gate", 4, 12, true);
    model.ffn_down = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.down", 12, 4, true);
    model.lm_head = pico_nn_linear_init(ctx, "tiny.lm_head", 4, 8, true);

    seed_param_values(ctx);
    return model;
}

static bool tiny_cpu_decoder_smoke_is_valid(struct TinyCpuDecoderSmoke* model) {
    return model != NULL && model->tok_emb != NULL && model->norm_local_0 != NULL && model->local_0 != NULL &&
           model->norm_local_1 != NULL && model->local_1 != NULL && model->norm_global != NULL &&
           model->global != NULL && model->norm_ffn != NULL && model->ffn_up != NULL &&
           model->ffn_gate != NULL && model->ffn_down != NULL && model->lm_head != NULL;
}

static struct PicoTensor* tiny_cpu_decoder_smoke_forward(
    struct PicoContext* ctx,
    struct TinyCpuDecoderSmoke* model,
    struct PicoTensor* tokens) {
    struct PicoTensor* h = pico_embedding_apply(ctx, model->tok_emb, tokens);
    struct PicoTensor* local = pico_nn_swa_attn_forward(ctx, model->local_0,
                                                        pico_nn_rmsnorm_forward(ctx, model->norm_local_0, h));
    h = pico_add(ctx, h, local);

    local = pico_nn_swa_attn_forward(ctx, model->local_1,
                                     pico_nn_rmsnorm_forward(ctx, model->norm_local_1, h));
    h = pico_add(ctx, h, local);

    struct PicoTensor* global = pico_nn_attn_forward(ctx, model->global,
                                                     pico_nn_rmsnorm_forward(ctx, model->norm_global, h));
    h = pico_add(ctx, h, global);

    struct PicoTensor* ffn_input = pico_nn_rmsnorm_forward(ctx, model->norm_ffn, h);
    struct PicoTensor* up = pico_nn_linear_forward(ctx, model->ffn_up, ffn_input);
    struct PicoTensor* gate = pico_nn_linear_forward(ctx, model->ffn_gate, ffn_input);
    struct PicoTensor* hidden = pico_swiglu(ctx, up, gate);
    struct PicoTensor* ffn_out = pico_nn_linear_forward(ctx, model->ffn_down, hidden);
    h = pico_add(ctx, h, ffn_out);

    return pico_nn_linear_forward(ctx, model->lm_head, h);
}

static void tiny_cpu_decoder_smoke_free(struct TinyCpuDecoderSmoke* model) {
    if(model == NULL) {
        return;
    }

    pico_nn_rmsnorm_free(model->norm_local_0);
    pico_nn_attn_free(model->local_0);
    pico_nn_rmsnorm_free(model->norm_local_1);
    pico_nn_attn_free(model->local_1);
    pico_nn_rmsnorm_free(model->norm_global);
    pico_nn_attn_free(model->global);
    pico_nn_rmsnorm_free(model->norm_ffn);
    pico_nn_linear_free(model->ffn_up);
    pico_nn_linear_free(model->ffn_gate);
    pico_nn_linear_free(model->ffn_down);
    pico_nn_linear_free(model->lm_head);
}

static bool any_grad_nonzero(struct PicoTensor* tensor) {
    if(tensor == NULL || tensor->grad == NULL) {
        return false;
    }

    for(int i = 0; i < tensor->numel; i++) {
        if(fabsf(tensor->grad[i]) > 1e-7f) {
            return true;
        }
    }

    return false;
}

UTEST(transformer, init_sets_child_modules_and_shapes) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 4, 2, 2, 8);
    ASSERT_TRUE(block != NULL);
    ASSERT_EQ(block->embed_dim, 4);
    ASSERT_EQ(block->num_heads, 2);
    ASSERT_EQ(block->d_k, 2);
    ASSERT_EQ(block->hidden_dim, 8);
    ASSERT_TRUE(block->attn != NULL);
    ASSERT_TRUE(block->ff1 != NULL);
    ASSERT_TRUE(block->ff2 != NULL);
    ASSERT_TRUE(block->ff1->weights->shape[0] == 4);
    ASSERT_TRUE(block->ff1->weights->shape[1] == 8);
    ASSERT_TRUE(block->ff2->weights->shape[0] == 8);
    ASSERT_TRUE(block->ff2->weights->shape[1] == 4);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, init_registers_named_child_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 4, 2, 2, 8);
    ASSERT_TRUE(block != NULL);
    ASSERT_TRUE(has_param_named(ctx, "block.attn.q_proj.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.attn.k_proj.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.attn.v_proj.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.attn.out_proj.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.ff1.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.ff1.bias"));
    ASSERT_TRUE(has_param_named(ctx, "block.ff2.weight"));
    ASSERT_TRUE(has_param_named(ctx, "block.ff2.bias"));

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, forward_2d_returns_same_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 4);
    ASSERT_TRUE(block != NULL);

    int64_t shape[] = {3, 2};
    float values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoTensor* out = pico_nn_transformer_forward(ctx, block, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_TRUE(out->shape[0] == 3);
    ASSERT_TRUE(out->shape[1] == 2);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, forward_3d_returns_same_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 4);
    ASSERT_TRUE(block != NULL);

    int64_t shape[] = {2, 3, 2};
    float values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        2.0f, 0.0f,
        0.0f, 2.0f,
        2.0f, 2.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 3, values);
    struct PicoTensor* out = pico_nn_transformer_forward(ctx, block, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 3);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 3);
    ASSERT_TRUE(out->shape[2] == 2);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, forward_zero_weights_behaves_like_residual_identity) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 4);
    ASSERT_TRUE(block != NULL);

    fill_zero(block->attn->Q);
    fill_zero(block->attn->K);
    fill_zero(block->attn->V);
    fill_zero(block->attn->O);
    fill_zero(block->ff1->weights);
    fill_zero(block->ff1->bias);
    fill_zero(block->ff2->weights);
    fill_zero(block->ff2->bias);

    int64_t shape[] = {2, 2};
    float values[] = {
        3.0f, -1.0f,
        2.0f, 4.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoTensor* out = pico_nn_transformer_forward(ctx, block, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR_FLOAT(out->data[0], 3.0f);
    ASSERT_NEAR_FLOAT(out->data[1], -1.0f);
    ASSERT_NEAR_FLOAT(out->data[2], 2.0f);
    ASSERT_NEAR_FLOAT(out->data[3], 4.0f);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, forward_uses_attention_and_mlp_residuals) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 2);
    ASSERT_TRUE(block != NULL);

    fill_identity(block->attn->Q);
    fill_identity(block->attn->K);
    fill_identity(block->attn->V);
    fill_zero(block->attn->O);
    block->attn->O->data[0] = 1.0f;

    fill_zero(block->ff1->weights);
    fill_zero(block->ff1->bias);
    fill_zero(block->ff2->weights);
    fill_zero(block->ff2->bias);
    block->ff1->weights->data[0] = 1.0f;
    block->ff2->weights->data[0] = 1.0f;

    int64_t shape[] = {1, 2};
    float values[] = {2.0f, 5.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoTensor* out = pico_nn_transformer_forward(ctx, block, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR_FLOAT(out->data[0], 8.0f);
    ASSERT_NEAR_FLOAT(out->data[1], 5.0f);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, backward_populates_attention_and_mlp_grads) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 2);
    ASSERT_TRUE(block != NULL);

    fill_identity(block->attn->Q);
    fill_identity(block->attn->K);
    fill_identity(block->attn->V);
    fill_identity(block->attn->O);
    fill_identity(block->ff1->weights);
    fill_zero(block->ff1->bias);
    fill_identity(block->ff2->weights);
    fill_zero(block->ff2->bias);

    int64_t shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float target_values[] = {
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, input_values);
    struct PicoTensor* target = pico_tensor_from_data(ctx, shape, 2, target_values);
    struct PicoTensor* out = pico_nn_transformer_forward(ctx, block, input);
    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(ctx, &mse, out, target);

    pico_backward(ctx, loss);

    ASSERT_TRUE(fabsf(block->attn->Q->grad[0]) > 0.0f || fabsf(block->attn->K->grad[0]) > 0.0f ||
                fabsf(block->attn->V->grad[0]) > 0.0f || fabsf(block->attn->O->grad[0]) > 0.0f);
    ASSERT_TRUE(fabsf(block->ff1->weights->grad[0]) > 0.0f);
    ASSERT_TRUE(fabsf(block->ff2->weights->grad[0]) > 0.0f);

    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, one_sgd_step_through_block_lowers_loss) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 2);
    ASSERT_TRUE(block != NULL);

    fill_identity(block->attn->Q);
    fill_identity(block->attn->K);
    fill_identity(block->attn->V);
    fill_identity(block->attn->O);
    fill_identity(block->ff1->weights);
    fill_zero(block->ff1->bias);
    fill_identity(block->ff2->weights);
    fill_zero(block->ff2->bias);

    int64_t shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float target_values[] = {
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, input_values);
    struct PicoTensor* target = pico_tensor_from_data(ctx, shape, 2, target_values);
    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoOptimSGD* sgd = pico_optim_sgd_init(0.01f);
    ASSERT_TRUE(sgd != NULL);

    struct PicoTensor* pred1 = pico_nn_transformer_forward(ctx, block, input);
    struct PicoTensor* loss1 = pico_mse_loss(ctx, &mse, pred1, target);
    float before = loss1->data[0];
    pico_backward(ctx, loss1);
    pico_optim_sgd_step(ctx, sgd);
    pico_optim_sgd_zero_grad(ctx, sgd);

    struct PicoTensor* pred2 = pico_nn_transformer_forward(ctx, block, input);
    struct PicoTensor* loss2 = pico_mse_loss(ctx, &mse, pred2, target);

    ASSERT_TRUE(loss2->data[0] < before);

    pico_optim_sgd_free(sgd);
    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, trains_for_multiple_steps_on_tiny_target) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTransformer* block = pico_nn_transformer_init(ctx, "block", 2, 1, 2, 2);
    ASSERT_TRUE(block != NULL);

    fill_identity(block->attn->Q);
    fill_identity(block->attn->K);
    fill_identity(block->attn->V);
    fill_identity(block->attn->O);
    fill_identity(block->ff1->weights);
    fill_zero(block->ff1->bias);
    fill_identity(block->ff2->weights);
    fill_zero(block->ff2->bias);

    int64_t shape[] = {2, 2};
    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    float target_values[] = {
        0.25f, 0.0f,
        0.0f, 0.25f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, input_values);
    struct PicoTensor* target = pico_tensor_from_data(ctx, shape, 2, target_values);
    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoOptimSGD* sgd = pico_optim_sgd_init(0.01f);
    ASSERT_TRUE(sgd != NULL);

    struct PicoTensor* pred = pico_nn_transformer_forward(ctx, block, input);
    struct PicoTensor* loss = pico_mse_loss(ctx, &mse, pred, target);
    float first_loss = loss->data[0];
    float last_loss = first_loss;

    for(int step = 0; step < 20; step++) {
        pred = pico_nn_transformer_forward(ctx, block, input);
        loss = pico_mse_loss(ctx, &mse, pred, target);
        last_loss = loss->data[0];

        pico_backward(ctx, loss);
        pico_optim_sgd_step(ctx, sgd);
        pico_optim_sgd_zero_grad(ctx, sgd);
    }

    pred = pico_nn_transformer_forward(ctx, block, input);
    loss = pico_mse_loss(ctx, &mse, pred, target);
    last_loss = loss->data[0];

    ASSERT_TRUE(last_loss < first_loss);
    ASSERT_TRUE(last_loss < first_loss * 0.8f);

    pico_optim_sgd_free(sgd);
    pico_nn_transformer_free(block);
    pico_shutdown(ctx);
}

UTEST(transformer, cpu_smoke_full_tiny_decoder_forward_and_backward) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct TinyCpuDecoderSmoke model = tiny_cpu_decoder_smoke_init(ctx);
    ASSERT_TRUE(tiny_cpu_decoder_smoke_is_valid(&model));

    int64_t token_shape[] = {4};
    float token_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float target_values[] = {2.0f, 3.0f, 4.0f, 5.0f};
    struct PicoTensor* tokens = pico_tensor_from_data(ctx, token_shape, 1, token_values);
    struct PicoTensor* targets = pico_tensor_from_data(ctx, token_shape, 1, target_values);
    struct PicoCrossEntropyLoss ce = {.reduction = PICO_CE_MEAN};

    struct PicoTensor* logits = tiny_cpu_decoder_smoke_forward(ctx, &model, tokens);
    ASSERT_TRUE(logits != NULL);
    ASSERT_EQ(logits->ndim, 2);
    ASSERT_TRUE(logits->shape[0] == 4);
    ASSERT_TRUE(logits->shape[1] == 8);

    struct PicoTensor* loss = pico_cross_entropy_loss(ctx, &ce, logits, targets);
    ASSERT_TRUE(loss != NULL);
    ASSERT_TRUE(isfinite(loss->data[0]));

    pico_backward(ctx, loss);

    ASSERT_TRUE(any_grad_nonzero(model.tok_emb->table));
    ASSERT_TRUE(any_grad_nonzero(model.local_0->Q) || any_grad_nonzero(model.local_0->K) ||
                any_grad_nonzero(model.local_0->V) || any_grad_nonzero(model.local_0->O));
    ASSERT_TRUE(any_grad_nonzero(model.local_1->Q) || any_grad_nonzero(model.local_1->K) ||
                any_grad_nonzero(model.local_1->V) || any_grad_nonzero(model.local_1->O));
    ASSERT_TRUE(any_grad_nonzero(model.global->Q) || any_grad_nonzero(model.global->K) ||
                any_grad_nonzero(model.global->V) || any_grad_nonzero(model.global->O));
    ASSERT_TRUE(any_grad_nonzero(model.ffn_down->weights));
    ASSERT_TRUE(any_grad_nonzero(model.lm_head->weights));

    tiny_cpu_decoder_smoke_free(&model);
    pico_shutdown(ctx);
}

UTEST(transformer, cpu_smoke_full_tiny_decoder_trains_one_batch) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct TinyCpuDecoderSmoke model = tiny_cpu_decoder_smoke_init(ctx);
    struct PicoOptimAdamW* optim = pico_optim_adamw_init(0.01f, 0.0f);
    ASSERT_TRUE(tiny_cpu_decoder_smoke_is_valid(&model));
    ASSERT_TRUE(optim != NULL);

    int64_t token_shape[] = {4};
    float token_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float target_values[] = {2.0f, 3.0f, 4.0f, 5.0f};
    struct PicoTensor* tokens = pico_tensor_from_data(ctx, token_shape, 1, token_values);
    struct PicoTensor* targets = pico_tensor_from_data(ctx, token_shape, 1, target_values);
    struct PicoCrossEntropyLoss ce = {.reduction = PICO_CE_MEAN};

    struct PicoTensor* logits = tiny_cpu_decoder_smoke_forward(ctx, &model, tokens);
    struct PicoTensor* loss = pico_cross_entropy_loss(ctx, &ce, logits, targets);
    ASSERT_TRUE(loss != NULL);
    float first_loss = loss->data[0];
    float last_loss = first_loss;

    for(int step = 0; step < 8; step++) {
        pico_optim_adamw_zero_grad(ctx, optim);

        logits = tiny_cpu_decoder_smoke_forward(ctx, &model, tokens);
        loss = pico_cross_entropy_loss(ctx, &ce, logits, targets);
        ASSERT_TRUE(loss != NULL);
        ASSERT_TRUE(isfinite(loss->data[0]));
        last_loss = loss->data[0];

        pico_backward(ctx, loss);
        pico_optim_adamw_step(ctx, optim);
    }

    logits = tiny_cpu_decoder_smoke_forward(ctx, &model, tokens);
    loss = pico_cross_entropy_loss(ctx, &ce, logits, targets);
    ASSERT_TRUE(loss != NULL);
    last_loss = loss->data[0];

    ASSERT_TRUE(isfinite(first_loss));
    ASSERT_TRUE(isfinite(last_loss));
    ASSERT_TRUE(last_loss < first_loss);

    pico_optim_adamw_free(optim);
    tiny_cpu_decoder_smoke_free(&model);
    pico_shutdown(ctx);
}
