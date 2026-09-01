#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "pico.h"
#include "safetensor/st.h"

static float rand_weight(void) {
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.4f;
}

static void randomize_tensor(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = rand_weight();
    }
}

static void init_layer_weights(struct PicoLinear* l1, struct PicoLinear* l2) {
    randomize_tensor(l1->weights);
    randomize_tensor(l1->bias);
    randomize_tensor(l2->weights);
    randomize_tensor(l2->bias);
}

static struct PicoTensor* forward(struct PicoContext* ctx, struct PicoLinear* l1, struct PicoLinear* l2,
                                  struct PicoTensor* x) {
    struct PicoTensor* h = pico_nn_linear_forward(ctx, l1, x);
    h = pico_relu(ctx, h);
    return pico_nn_linear_forward(ctx, l2, h);
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
    const char* file_name = "trained_relu_mlp.safetensors";

    srand(7);

    struct PicoContext* ctx = pico_init();
    if(ctx == NULL) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    int64_t x_shape[] = {8, 2};
    int64_t y_shape[] = {8, 1};
    float x_values[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        2.0f, 0.0f,
        0.0f, 2.0f,
        2.0f, 1.0f,
        1.0f, 2.0f,
    };
    float y_values[] = {
        1.0f,
        4.0f,
        3.0f,
        6.0f,
        5.0f,
        7.0f,
        8.0f,
        9.0f,
    };

    struct PicoTensor* x = pico_tensor_from_data(ctx, x_shape, 2, x_values);
    struct PicoTensor* y = pico_tensor_from_data(ctx, y_shape, 2, y_values);

    struct PicoLinear* l1 = pico_nn_linear_init(ctx, "mlp.l1", 2, 8, true);
    struct PicoLinear* l2 = pico_nn_linear_init(ctx, "mlp.l2", 8, 1, true);
    if(x == NULL || y == NULL || l1 == NULL || l2 == NULL) {
        fprintf(stderr, "failed to build training example\n");
        pico_nn_linear_free(l1);
        pico_nn_linear_free(l2);
        pico_shutdown(ctx);
        return 1;
    }

    init_layer_weights(l1, l2);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    struct PicoMSELoss mse = {.reduction = MEAN};

    printf("training: y = 1 + 2*x0 + 3*x1\n");
    printf("model: Linear(2,8) -> ReLU -> Linear(8,1)\n\n");

    for(int step = 0; step <= 1000; step++) {
        struct PicoTensor* pred = forward(ctx, l1, l2, x);
        struct PicoTensor* loss = pico_mse_loss(ctx, &mse, pred, y);

        if(step % 100 == 0) {
            printf("step %4d | loss %.6f | pred[0] %.4f | target[0] %.4f\n", step,
                   loss->data[0], pred->data[0], y->data[0]);
        }

        pico_optim_sgd_zero_grad(ctx, opt);
        pico_backward(ctx, loss);
        pico_optim_sgd_step(ctx, opt);
    }

    struct PicoTensor* final_pred = forward(ctx, l1, l2, x);
    struct PicoTensor* final_loss = pico_mse_loss(ctx, &mse, final_pred, y);

    printf("\nfinal loss: %.6f\n", final_loss->data[0]);
    printf("\nfinal predictions:\n");
    pico_tensor_print(final_pred);

    save_tensor(ctx, (char*)file_name);
    printf("\nsaved trained weights to %s\n", file_name);
    print_param_summary(ctx);
    print_safetensor_header(file_name);

    pico_optim_sgd_free(opt);
    pico_nn_linear_free(l1);
    pico_nn_linear_free(l2);
    pico_shutdown(ctx);

    return 0;
}
