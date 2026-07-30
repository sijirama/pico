#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico.h"
#include "tokens/wordbased-tk.h"

struct TextDatasetData {
    struct PicoContext* ctx;
    struct Tokenizer* tokenizer;
    char** texts;
    float* labels;
    size_t len;
};

static void free_wordbased_map(struct Tokenizer* tokenizer) {
    if(tokenizer == NULL || tokenizer->data == NULL) {
        return;
    }

    struct WordBasedPicoTKData* data = (struct WordBasedPicoTKData*)tokenizer->data;
    pico_hashmap_free(data->word_to_id_map);
    data->word_to_id_map = NULL;
}

static size_t text_dataset_len(const struct Dataset* dataset) {
    struct TextDatasetData* data = (struct TextDatasetData*)dataset->data;
    return data->len;
}

static struct DatasetItem text_dataset_get(const struct Dataset* dataset, size_t idx) {
    struct TextDatasetData* data = (struct TextDatasetData*)dataset->data;
    size_t* ids = data->tokenizer->methods->encode(data->tokenizer, data->texts[idx]);

    size_t token_count = 0;
    while(ids[token_count] != (size_t)-1) {
        token_count++;
    }

    float* token_values = arena_alloc(data->ctx->arena, sizeof(float) * token_count);
    for(size_t i = 0; i < token_count; i++) {
        token_values[i] = (float)ids[i];
    }

    int64_t x_shape[] = {(int64_t)token_count};
    int64_t y_shape[] = {1};
    float y_value[] = {data->labels[idx]};

    struct DatasetItem item = {
        .x = pico_tensor_from_data(data->ctx, x_shape, 1, token_values),
        .y = pico_tensor_from_data(data->ctx, y_shape, 1, y_value),
    };
    return item;
}

static void text_dataset_free(struct Dataset* dataset) {
    (void)dataset;
}

static const struct DatasetVTable TEXT_DATASET_FUNCS = {
    .len = text_dataset_len,
    .get = text_dataset_get,
    .free = text_dataset_free,
};

static void strip_newline(char* line) {
    size_t len = strlen(line);
    while(len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        len--;
    }
}

static char* arena_strdup(struct PicoContext* ctx, const char* text) {
    size_t len = strlen(text);
    char* copy = arena_alloc(ctx->arena, len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static void add_text_to_vocab(struct PicoContext* ctx, struct Tokenizer* tokenizer, const char* text) {
    char* text_copy = arena_strdup(ctx, text);
    if(text_copy == NULL) {
        return;
    }

    char* token = strtok(text_copy, " \t\r\n");
    while(token != NULL) {
        pico_wordbased_add_word(tokenizer, token);
        token = strtok(NULL, " \t\r\n");
    }
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

static struct Dataset text_dataset_from_file(struct PicoContext* ctx, struct Tokenizer* tokenizer, const char* path) {
    struct Dataset dataset = {0};
    FILE* file = fopen(path, "r");
    if(file == NULL) {
        return dataset;
    }

    size_t rows = count_file_rows(file);
    struct TextDatasetData* data = arena_alloc(ctx->arena, sizeof(struct TextDatasetData));
    char** texts = arena_alloc(ctx->arena, sizeof(char*) * rows);
    float* labels = arena_alloc(ctx->arena, sizeof(float) * rows);
    if(data == NULL || texts == NULL || labels == NULL) {
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

        char* tab = strchr(line, '\t');
        if(tab == NULL) {
            continue;
        }

        *tab = '\0';
        char* text = tab + 1;

        labels[row] = strtof(line, NULL);
        texts[row] = arena_strdup(ctx, text);
        if(texts[row] == NULL) {
            fclose(file);
            return dataset;
        }

        add_text_to_vocab(ctx, tokenizer, texts[row]);
        row++;
    }

    fclose(file);

    data->ctx = ctx;
    data->tokenizer = tokenizer;
    data->texts = texts;
    data->labels = labels;
    data->len = row;

    dataset.funcs = &TEXT_DATASET_FUNCS;
    dataset.data = data;
    return dataset;
}

int main(void) {
    pico_init();
    struct PicoContext ctx = pico_context_init();

    struct Tokenizer* tokenizer = pico_wordbased_create_init(&ctx);
    if(tokenizer == NULL) {
        fprintf(stderr, "failed to create tokenizer\n");
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    struct Dataset dataset =
        text_dataset_from_file(&ctx, tokenizer, "examples/03_tokenizer_dataloader/data.txt");
    if(dataset.funcs == NULL) {
        fprintf(stderr, "failed to load text dataset\n");
        free_wordbased_map(tokenizer);
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);

    printf("vocab size: %zu\n", tokenizer->methods->len(tokenizer));

    struct DataBatch* batch = NULL;
    while((batch = pico_dataloader_next(loader)) != NULL) {
        printf("batch size: %zu\n", batch->size);
        for(size_t i = 0; i < batch->size; i++) {
            pico_tensor_print(batch->items[i].x);
            pico_tensor_print(batch->items[i].y);
        }
    }

    dataset.funcs->free(&dataset);
    free_wordbased_map(tokenizer);
    pico_context_destroy(&ctx);
    pico_shutdown();
    return 0;
}
