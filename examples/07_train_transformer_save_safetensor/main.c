#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "pico.h"
#include "safetensor/st.h"

static float rand_weight(void) {
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
}

static void randomize_param(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = rand_weight();
    }
}

static void randomize_context_params(struct PicoContext* ctx) {
    for(size_t i = 0; i < ctx->params.size; i++) {
        randomize_param(ctx->params.data[i]);
    }
}

static struct PicoTensor* transformer_forward(struct PicoContext* ctx, struct PicoTransformer* block1,
                                              struct PicoTransformer* block2, struct PicoTensor* x) {
    struct PicoTensor* h = pico_nn_transformer_forward(ctx, block1, x);
    if(h == NULL) {
        return NULL;
    }

    return pico_nn_transformer_forward(ctx, block2, h);
}

static void print_param_summary(struct PicoContext* ctx) {
    printf("\nsaved params:\n");
    for(size_t i = 0; i < ctx->params.size; i++) {
        struct PicoTensor* param = ctx->params.data[i];
        printf("  %s shape=[", param->name);
        for(int dim = 0; dim < param->ndim; dim++) {
            printf("%ld%s", (long)param->shape[dim], dim + 1 == param->ndim ? "" : ", ");
        }
        printf("] numel=%ld\n", (long)param->numel);
    }
}

static void print_safetensor_header(const char* file_name) {
    FILE* file = fopen(file_name, "rb");
    if(file == NULL) {
        fprintf(stderr, "could not open saved safetensors file\n");
        return;
    }

    uint64_t header_size = 0;
    if(fread(&header_size, sizeof(uint64_t), 1, file) != 1) {
        fclose(file);
        return;
    }

    char* header = malloc(header_size + 1);
    if(header == NULL) {
        fclose(file);
        return;
    }

    fread(header, 1, header_size, file);
    header[header_size] = '\0';

    printf("\nsafetensors header (%lu bytes):\n%s\n", (unsigned long)header_size, header);

    free(header);
    fclose(file);
}

int main(void) {
    const char* file_name = "tiny_transformer.safetensors";

    srand(11);

    struct PicoContext* ctx = pico_init();
    if(ctx == NULL) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    int64_t shape[] = {3, 4};
    float x_values[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    float y_values[] = {
        0.25f, 0.00f, 0.00f, 0.00f,
        0.00f, 0.25f, 0.00f, 0.00f,
        0.00f, 0.00f, 0.25f, 0.00f,
    };

    struct PicoTensor* x = pico_tensor_from_data(ctx, shape, 2, x_values);
    struct PicoTensor* y = pico_tensor_from_data(ctx, shape, 2, y_values);
    struct PicoTransformer* block1 = pico_nn_transformer_init(ctx, "transformer.blocks.0", 4, 2, 2, 8);
    struct PicoTransformer* block2 = pico_nn_transformer_init(ctx, "transformer.blocks.1", 4, 2, 2, 8);

    if(x == NULL || y == NULL || block1 == NULL || block2 == NULL) {
        fprintf(stderr, "failed to build transformer example\n");
        pico_nn_transformer_free(block1);
        pico_nn_transformer_free(block2);
        pico_shutdown(ctx);
        return 1;
    }

    randomize_context_params(ctx);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    struct PicoMSELoss mse = {.reduction = MEAN};

    printf("training tiny transformer\n");
    printf("model: 2 x [causal mha + residual + linear -> relu -> linear + residual]\n");
    printf("input shape: [seq=3, embed=4]\n\n");

    for(int step = 0; step <= 100; step++) {
        struct PicoTensor* pred = transformer_forward(ctx, block1, block2, x);
        struct PicoTensor* loss = pico_mse_loss(ctx, &mse, pred, y);

        if(step % 20 == 0) {
            printf("step %3d | loss %.6f | pred[0] %.4f | target[0] %.4f\n", step,
                   loss->data[0], pred->data[0], y->data[0]);
        }

        pico_optim_sgd_zero_grad(ctx, opt);
        pico_backward(ctx, loss);
        pico_optim_sgd_step(ctx, opt);
    }

    struct PicoTensor* final_pred = transformer_forward(ctx, block1, block2, x);
    struct PicoTensor* final_loss = pico_mse_loss(ctx, &mse, final_pred, y);

    printf("\nfinal loss: %.6f\n", final_loss->data[0]);
    printf("\nfinal predictions:\n");
    pico_tensor_print(final_pred);

    save_tensor(ctx, (char*)file_name);
    printf("\nsaved trained transformer weights to %s\n", file_name);
    print_param_summary(ctx);
    print_safetensor_header(file_name);

    pico_optim_sgd_free(opt);
    pico_nn_transformer_free(block1);
    pico_nn_transformer_free(block2);
    pico_shutdown(ctx);

    return 0;
}
