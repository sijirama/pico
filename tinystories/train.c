#include "main.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "act/activations.h"
#include "arena.h"
#include "loss/loss.h"
#include "nn/attn.h"
#include "nn/embedding.h"
#include "nn/linear.h"
#include "norm/norm.h"
#include "optim/optim.h"
#include "safetensor/st.h"
#include "tokens/bpe-tk.h"

#define TINYSTORIES_SMOKE_STEPS 12

struct TinyStoriesCpuSmokeModel {
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

static bool set_tensor_name(struct PicoContext* ctx, struct PicoTensor* tensor, const char* name) {
    if(ctx == NULL || tensor == NULL || name == NULL) {
        return false;
    }

    struct Arena* arena = pico_context_param_arena(ctx);
    if(arena == NULL) {
        return false;
    }

    size_t len = strlen(name);
    tensor->name = arena_alloc(arena, len + 1);
    if(tensor->name == NULL) {
        return false;
    }

    memcpy(tensor->name, name, len + 1);
    return true;
}

static void seed_model_params(struct PicoContext* ctx) {
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
                int bucket = ((p + 3) * (i + 5)) % 17;
                tensor->data[i] = ((float)bucket - 8.0f) * 0.015f;
            }
        }
    }
}

static struct TinyStoriesCpuSmokeModel model_init(struct PicoContext* ctx, int vocab_size) {
    struct TinyStoriesCpuSmokeModel model = {0};

    model.tok_emb = pico_embedding_init(ctx, vocab_size, 8);
    if(model.tok_emb != NULL) {
        set_tensor_name(ctx, model.tok_emb->table, "tiny.embed.weight");
    }

    model.norm_local_0 = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.local.0.norm", 8, 1e-5f);
    model.local_0 = pico_nn_swa_attn_init(ctx, "tiny.blocks.0.local.0.attn", 8, 2, 4, 4);
    model.norm_local_1 = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.local.1.norm", 8, 1e-5f);
    model.local_1 = pico_nn_swa_attn_init(ctx, "tiny.blocks.0.local.1.attn", 8, 2, 4, 4);
    model.norm_global = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.global.norm", 8, 1e-5f);
    model.global = pico_nn_attn_init(ctx, "tiny.blocks.0.global.attn", 8, 2, 4);
    model.norm_ffn = pico_nn_rmsnorm_init(ctx, "tiny.blocks.0.ffn.norm", 8, 1e-5f);
    model.ffn_up = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.up", 8, 24, true);
    model.ffn_gate = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.gate", 8, 24, true);
    model.ffn_down = pico_nn_linear_init(ctx, "tiny.blocks.0.ffn.down", 24, 8, true);
    model.lm_head = pico_nn_linear_init(ctx, "tiny.lm_head", 8, vocab_size, true);

    seed_model_params(ctx);
    return model;
}

static bool model_is_valid(struct TinyStoriesCpuSmokeModel* model) {
    return model != NULL && model->tok_emb != NULL && model->norm_local_0 != NULL && model->local_0 != NULL &&
           model->norm_local_1 != NULL && model->local_1 != NULL && model->norm_global != NULL &&
           model->global != NULL && model->norm_ffn != NULL && model->ffn_up != NULL &&
           model->ffn_gate != NULL && model->ffn_down != NULL && model->lm_head != NULL;
}

static struct PicoTensor* model_forward(
    struct PicoContext* ctx,
    struct TinyStoriesCpuSmokeModel* model,
    struct PicoTensor* tokens) {
    struct PicoTensor* h = pico_embedding_apply(ctx, model->tok_emb, tokens);
    if(h == NULL) {
        return NULL;
    }

    struct PicoTensor* x = pico_nn_rmsnorm_forward(ctx, model->norm_local_0, h);
    struct PicoTensor* local = pico_nn_swa_attn_forward(ctx, model->local_0, x);
    h = pico_add(ctx, h, local);

    x = pico_nn_rmsnorm_forward(ctx, model->norm_local_1, h);
    local = pico_nn_swa_attn_forward(ctx, model->local_1, x);
    h = pico_add(ctx, h, local);

    x = pico_nn_rmsnorm_forward(ctx, model->norm_global, h);
    struct PicoTensor* global = pico_nn_attn_forward(ctx, model->global, x);
    h = pico_add(ctx, h, global);

    x = pico_nn_rmsnorm_forward(ctx, model->norm_ffn, h);
    struct PicoTensor* up = pico_nn_linear_forward(ctx, model->ffn_up, x);
    struct PicoTensor* gate = pico_nn_linear_forward(ctx, model->ffn_gate, x);
    struct PicoTensor* hidden = pico_swiglu(ctx, up, gate);
    struct PicoTensor* ffn_out = pico_nn_linear_forward(ctx, model->ffn_down, hidden);
    h = pico_add(ctx, h, ffn_out);

    return pico_nn_linear_forward(ctx, model->lm_head, h);
}

static void model_free(struct TinyStoriesCpuSmokeModel* model) {
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

static void print_param_summary(struct PicoContext* ctx) {
    printf("\nsaved params: %ld\n", (long)ctx->params.size);
    for(int i = 0; i < ctx->params.size; i++) {
        struct PicoTensor* param = ctx->params.data[i];
        printf("  %s shape=[", param->name == NULL ? "<unnamed>" : param->name);
        for(int d = 0; d < param->ndim; d++) {
            printf("%ld%s", (long)param->shape[d], d + 1 == param->ndim ? "" : ", ");
        }
        printf("] numel=%ld\n", (long)param->numel);
    }
}

int main(void) {
    const char* save_path = "tinystories_cpu_smoke.safetensors";
    struct PicoContext* ctx = pico_init_verbose(false);
    if(ctx == NULL) {
        fprintf(stderr, "failed to create pico context\n");
        return 1;
    }

    struct TinyStoriesDatasetConfig config = tinystories_default_config();
    config.max_rows = 128;
    config.batch_size = 1;
    config.shuffle = false;
    config.max_vocab_size = 256;
    config.max_seq_len = 16;

    struct TinyStoriesDataset dataset = {0};
    if(!tinystories_dataset_prepare(ctx, &dataset, config)) {
        pico_shutdown(ctx);
        return 1;
    }

    int vocab_size = (int)dataset.tokenizer->methods->len(dataset.tokenizer);
    printf("tinystories cpu smoke\n");
    printf("dataset: %s\n", config.path);
    printf("rows: %ld | vocab: %d | seq_len: %d | steps: %d\n\n",
           (long)dataset.len, vocab_size, config.max_seq_len, TINYSTORIES_SMOKE_STEPS);

    struct TinyStoriesCpuSmokeModel model = model_init(ctx, vocab_size);
    if(!model_is_valid(&model)) {
        fprintf(stderr, "failed to create tiny stories smoke model\n");
        tinystories_dataset_free(&dataset);
        pico_shutdown(ctx);
        return 1;
    }

    struct PicoOptimAdamW* optim = pico_optim_adamw_init(0.005f, 0.0f);
    struct PicoCrossEntropyLoss ce = {.reduction = PICO_CE_MEAN};
    if(optim == NULL) {
        model_free(&model);
        tinystories_dataset_free(&dataset);
        pico_shutdown(ctx);
        return 1;
    }

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for(int step = 0; step < TINYSTORIES_SMOKE_STEPS; step++) {
        struct DataBatch* batch = pico_dataloader_next(dataset.loader);
        if(batch == NULL || batch->size == 0 || batch->items[0].x == NULL || batch->items[0].y == NULL) {
            tinystories_dataset_reset(&dataset);
            batch = pico_dataloader_next(dataset.loader);
        }
        if(batch == NULL || batch->size == 0 || batch->items[0].x == NULL || batch->items[0].y == NULL) {
            fprintf(stderr, "failed to get tinystories batch\n");
            break;
        }

        struct PicoTensor* logits = model_forward(ctx, &model, batch->items[0].x);
        struct PicoTensor* loss = pico_cross_entropy_loss(ctx, &ce, logits, batch->items[0].y);
        if(loss == NULL || !isfinite(loss->data[0])) {
            fprintf(stderr, "training produced invalid loss\n");
            break;
        }

        if(step == 0) {
            first_loss = loss->data[0];
        }
        last_loss = loss->data[0];

        printf("step %02d | loss %.6f | tokens %ld\n", step, loss->data[0], (long)batch->items[0].x->numel);

        pico_optim_adamw_zero_grad(ctx, optim);
        pico_backward(ctx, loss);
        pico_optim_adamw_step(ctx, optim);
    }

    printf("\nfirst loss: %.6f\n", first_loss);
    printf("last loss:  %.6f\n", last_loss);

    save_tensor(ctx, (char*)save_path);
    printf("\nsaved smoke checkpoint to %s\n", save_path);
    print_param_summary(ctx);

    pico_optim_adamw_free(optim);
    model_free(&model);
    tinystories_dataset_free(&dataset);
    pico_shutdown(ctx);
    return 0;
}
