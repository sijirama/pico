#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico.h"
#include "tokens/bpe-tk.h"

struct NextTokenTextDatasetData {
    struct PicoContext* ctx;
    struct Tokenizer* tokenizer;
    char** texts;
    size_t len;
};

static void free_bpe_example_maps(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    struct BPEPicoTKData* data = (struct BPEPicoTKData*)tokenizer->data;
    pico_hashmap_free(data->corpus);
    pico_hashmap_free(data->token_to_id);
    if(data->vocab != NULL) {
        pico_vec_free(data->vocab);
        free(data->vocab);
    }
    if(data->merges != NULL) {
        pico_vec_free(data->merges);
        free(data->merges);
    }

    data->corpus = NULL;
    data->token_to_id = NULL;
    data->vocab = NULL;
    data->merges = NULL;
}

static void strip_newline(char* line) {
    size_t len = strlen(line);
    while(len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        len--;
    }
}

static char* example_arena_strdup(struct PicoContext* ctx, const char* text) {
    size_t len = strlen(text);
    char* copy = arena_alloc(ctx->arena, len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static size_t count_file_rows(FILE* file) {
    char line[512];
    size_t rows = 0;

    while(fgets(line, sizeof(line), file) != NULL) {
        strip_newline(line);
        if(line[0] != '\0') {
            rows++;
        }
    }

    rewind(file);
    return rows;
}

static size_t next_token_dataset_len(const struct Dataset* dataset) {
    struct NextTokenTextDatasetData* data = (struct NextTokenTextDatasetData*)dataset->data;
    return data->len;
}

static struct DatasetItem next_token_dataset_get(const struct Dataset* dataset, size_t idx) {
    struct NextTokenTextDatasetData* data = (struct NextTokenTextDatasetData*)dataset->data;
    size_t* ids = data->tokenizer->methods->encode(data->tokenizer, data->texts[idx]);

    size_t token_count = 0;
    while(ids[token_count] != (size_t)-1) {
        token_count++;
    }

    size_t sample_len = token_count > 1 ? token_count - 1 : 1;
    float* x_values = arena_alloc(data->ctx->arena, sizeof(float) * sample_len);
    float* y_values = arena_alloc(data->ctx->arena, sizeof(float) * sample_len);

    for(size_t i = 0; i < sample_len; i++) {
        x_values[i] = (float)ids[i];
        y_values[i] = (float)ids[i + 1];
    }

    int64_t shape[] = {(int64_t)sample_len};
    struct DatasetItem item = {
        .x = pico_tensor_from_data(data->ctx, shape, 1, x_values),
        .y = pico_tensor_from_data(data->ctx, shape, 1, y_values),
    };
    return item;
}

static void next_token_dataset_free(struct Dataset* dataset) {
    (void)dataset;
}

static const struct DatasetVTable NEXT_TOKEN_DATASET_FUNCS = {
    .len = next_token_dataset_len,
    .get = next_token_dataset_get,
    .free = next_token_dataset_free,
};

static struct Dataset next_token_dataset_from_file(struct PicoContext* ctx, struct Tokenizer* tokenizer,
                                                   const char* path) {
    struct Dataset dataset = {0};
    FILE* file = fopen(path, "r");
    if(file == NULL) {
        return dataset;
    }

    size_t rows = count_file_rows(file);
    struct NextTokenTextDatasetData* data = arena_alloc(ctx->arena, sizeof(struct NextTokenTextDatasetData));
    char** texts = arena_alloc(ctx->arena, sizeof(char*) * rows);
    if(data == NULL || texts == NULL) {
        fclose(file);
        return dataset;
    }

    char line[512];
    size_t row = 0;
    while(row < rows && fgets(line, sizeof(line), file) != NULL) {
        strip_newline(line);
        if(line[0] == '\0') {
            continue;
        }

        texts[row] = example_arena_strdup(ctx, line);
        if(texts[row] == NULL) {
            fclose(file);
            return dataset;
        }

        char* train_copy = example_arena_strdup(ctx, line);
        if(train_copy != NULL) {
            bpe_ingest_text(tokenizer, train_copy);
        }
        row++;
    }

    fclose(file);

    data->ctx = ctx;
    data->tokenizer = tokenizer;
    data->texts = texts;
    data->len = row;

    dataset.funcs = &NEXT_TOKEN_DATASET_FUNCS;
    dataset.data = data;
    return dataset;
}

int main(void) {
    pico_init();
    struct PicoContext ctx = pico_context_init();

    struct Tokenizer* tokenizer = pico_bpe_tk_init(&ctx);
    if(tokenizer == NULL) {
        fprintf(stderr, "failed to create bpe tokenizer\n");
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    struct BPEPicoTKData* bpe_data = (struct BPEPicoTKData*)tokenizer->data;
    bpe_data->max_vocab_capacity = 64;

    struct Dataset dataset = next_token_dataset_from_file(&ctx, tokenizer, "data.txt");
    if(dataset.funcs == NULL) {
        fprintf(stderr, "failed to load next-token dataset\n");
        free_bpe_example_maps(tokenizer);
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    bpe_train(tokenizer);

    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);
    if(loader == NULL) {
        fprintf(stderr, "failed to create dataloader\n");
        dataset.funcs->free(&dataset);
        free_bpe_example_maps(tokenizer);
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    printf("bpe vocab size: %zu\n", tokenizer->methods->len(tokenizer));
    printf("dataset item shape: x = tokens[:-1], y = tokens[1:]\n\n");

    struct DataBatch* batch = NULL;
    while((batch = pico_dataloader_next(loader)) != NULL) {
        printf("batch size: %zu\n", batch->size);
        for(size_t i = 0; i < batch->size; i++) {
            printf("x token ids:\n");
            pico_tensor_print(batch->items[i].x);
            printf("y shifted token ids:\n");
            pico_tensor_print(batch->items[i].y);
        }
    }

    dataset.funcs->free(&dataset);
    free_bpe_example_maps(tokenizer);
    pico_context_destroy(&ctx);
    pico_shutdown();
    return 0;
}
