#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "pico.h"
#include "safetensor/st.h"

static float rand_weight(void) {
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
}

static void randomize_tensor(struct PicoTensor* tensor) {
    for(int i = 0; i < tensor->numel; i++) {
        tensor->data[i] = rand_weight();
    }
}

static void init_mlp_random(struct PicoLinear* l1, struct PicoLinear* l2) {
    randomize_tensor(l1->weights);
    randomize_tensor(l1->bias);
    randomize_tensor(l2->weights);
    randomize_tensor(l2->bias);
}

static void print_saved_header(const char* file_name) {
    FILE* file = fopen(file_name, "rb");
    if(file == NULL) {
        fprintf(stderr, "could not reopen saved safetensors file\n");
        return;
    }

    uint64_t header_size = 0;
    fread(&header_size, sizeof(uint64_t), 1, file);

    char* header = malloc(header_size + 1);
    if(header == NULL) {
        fclose(file);
        return;
    }

    fread(header, 1, header_size, file);
    header[header_size] = '\0';

    printf("\nsaved header size: %lu bytes\n", (unsigned long)header_size);
    printf("\nsafetensors header:\n%s\n", header);

    free(header);
    fclose(file);
}

int main(void) {
    const char* file_name = "mlp_random.safetensors";

    srand(42);

    struct PicoContext* ctx = pico_init();
    if(ctx == NULL) {
        fprintf(stderr, "failed to initialize pico\n");
        return 1;
    }

    struct PicoLinear* l1 = pico_nn_linear_init(ctx, "mlp.l1", 4, 16, true);
    struct PicoLinear* l2 = pico_nn_linear_init(ctx, "mlp.l2", 16, 3, true);
    if(l1 == NULL || l2 == NULL) {
        fprintf(stderr, "failed to create mlp params\n");
        pico_nn_linear_free(l1);
        pico_nn_linear_free(l2);
        pico_shutdown(ctx);
        return 1;
    }

    init_mlp_random(l1, l2);

    printf("created random 2-layer mlp\n");
    printf("architecture: Linear(4, 16) -> ReLU -> Linear(16, 3)\n");
    printf("ctx params: %zu\n", ctx->params.size);

    for(size_t i = 0; i < ctx->params.size; i++) {
        struct PicoTensor* param = ctx->params.data[i];
        printf("  %s shape=[", param->name);
        for(int dim = 0; dim < param->ndim; dim++) {
            printf("%ld%s", (long)param->shape[dim], dim + 1 == param->ndim ? "" : ", ");
        }
        printf("] numel=%ld\n", (long)param->numel);
    }

    save_tensor(ctx, (char*)file_name);
    printf("\nsaved weights to %s\n", file_name);
    print_saved_header(file_name);

    pico_nn_linear_free(l1);
    pico_nn_linear_free(l2);
    pico_shutdown(ctx);

    return 0;
}
