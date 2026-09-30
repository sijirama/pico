#include "main.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "tokens/bpe-tk.h"

static void tinystories_strip_newline(char* line) {
    size_t len = strlen(line);
    while(len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[len - 1] = '\0';
        len--;
    }
}

static char* tinystories_arena_strdup(struct PicoContext* ctx, const char* text) {
    size_t len = strlen(text);
    char* copy = arena_alloc(ctx->arena, len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static size_t tinystories_count_rows(FILE* file, size_t max_rows) {
    char line[4096];
    size_t rows = 0;

    while(fgets(line, sizeof(line), file) != NULL) {
        tinystories_strip_newline(line);
        if(line[0] == '\0') {
            continue;
        }

        rows++;
        if(max_rows > 0 && rows >= max_rows) {
            break;
        }
    }

    rewind(file);
    return rows;
}

static size_t tinystories_dataset_len(const struct Dataset* dataset) {
    const struct TinyStoriesDataset* data = (const struct TinyStoriesDataset*)dataset->data;
    return data == NULL ? 0 : data->len;
}

static struct DatasetItem tinystories_dataset_get(const struct Dataset* dataset, size_t idx) {
    struct DatasetItem item = {0};
    struct TinyStoriesDataset* data = (struct TinyStoriesDataset*)dataset->data;
    if(data == NULL || data->tokenizer == NULL || idx >= data->len) {
        return item;
    }

    size_t* ids = data->tokenizer->methods->encode(data->tokenizer, data->texts[idx]);
    if(ids == NULL) {
        return item;
    }

    size_t token_count = 0;
    while(ids[token_count] != (size_t)-1) {
        token_count++;
    }

    if(token_count < 2) {
        return item;
    }

    size_t sample_len = token_count - 1;
    if(data->max_seq_len > 0 && sample_len > (size_t)data->max_seq_len) {
        sample_len = (size_t)data->max_seq_len;
    }

    float* x_values = arena_alloc(data->ctx->arena, sizeof(float) * sample_len);
    float* y_values = arena_alloc(data->ctx->arena, sizeof(float) * sample_len);
    if(x_values == NULL || y_values == NULL) {
        return item;
    }

    for(size_t i = 0; i < sample_len; i++) {
        x_values[i] = (float)ids[i];
        y_values[i] = (float)ids[i + 1];
    }

    int64_t shape[] = {(int64_t)sample_len};
    item.x = pico_tensor_from_data(data->ctx, shape, 1, x_values);
    item.y = pico_tensor_from_data(data->ctx, shape, 1, y_values);
    return item;
}

static void tinystories_dataset_vtable_free(struct Dataset* dataset) {
    (void)dataset;
}

static const struct DatasetVTable TINYSTORIES_DATASET_FUNCS = {
    .len = tinystories_dataset_len,
    .get = tinystories_dataset_get,
    .free = tinystories_dataset_vtable_free,
};

static void tinystories_free_bpe_heap_state(struct Tokenizer* tokenizer) {
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

struct TinyStoriesDatasetConfig tinystories_default_config(void) {
    struct TinyStoriesDatasetConfig config = {
        .path = TINYSTORIES_DEFAULT_TRAIN_PATH,
        .max_rows = 512,
        .batch_size = 1,
        .shuffle = true,
        .max_vocab_size = 1024,
        .max_seq_len = 32,
    };
    return config;
}

bool tinystories_dataset_prepare(struct PicoContext* ctx, struct TinyStoriesDataset* out,
                                 struct TinyStoriesDatasetConfig config) {
    if(ctx == NULL || out == NULL || config.path == NULL || config.batch_size == 0) {
        return false;
    }

    memset(out, 0, sizeof(*out));
    out->ctx = ctx;
    out->max_seq_len = config.max_seq_len;

    FILE* file = fopen(config.path, "r");
    if(file == NULL) {
        fprintf(stderr, "TinyStoriesDatasetError: could not open %s\n", config.path);
        return false;
    }

    size_t rows = tinystories_count_rows(file, config.max_rows);
    if(rows == 0) {
        fclose(file);
        fprintf(stderr, "TinyStoriesDatasetError: no rows found in %s\n", config.path);
        return false;
    }

    out->texts = arena_alloc(ctx->arena, sizeof(char*) * rows);
    if(out->texts == NULL) {
        fclose(file);
        return false;
    }

    out->tokenizer = pico_bpe_tk_init(ctx);
    if(out->tokenizer == NULL) {
        fclose(file);
        return false;
    }

    struct BPEPicoTKData* bpe_data = (struct BPEPicoTKData*)out->tokenizer->data;
    if(config.max_vocab_size > 0) {
        bpe_data->max_vocab_capacity = config.max_vocab_size;
    }

    char line[4096];
    size_t row = 0;
    while(row < rows && fgets(line, sizeof(line), file) != NULL) {
        tinystories_strip_newline(line);
        if(line[0] == '\0') {
            continue;
        }

        out->texts[row] = tinystories_arena_strdup(ctx, line);
        char* train_copy = tinystories_arena_strdup(ctx, line);
        if(out->texts[row] == NULL || train_copy == NULL) {
            fclose(file);
            return false;
        }

        bpe_ingest_text(out->tokenizer, train_copy);
        row++;
    }

    fclose(file);

    out->len = row;
    bpe_train(out->tokenizer);

    out->dataset.funcs = &TINYSTORIES_DATASET_FUNCS;
    out->dataset.data = out;
    out->loader = pico_dataloader_init(ctx, &out->dataset, config.batch_size, config.shuffle);
    if(out->loader == NULL) {
        return false;
    }

    return true;
}

void tinystories_dataset_reset(struct TinyStoriesDataset* dataset) {
    if(dataset == NULL || dataset->loader == NULL) {
        return;
    }

    pico_dataloader_reset(dataset->loader);
}

void tinystories_dataset_free(struct TinyStoriesDataset* dataset) {
    if(dataset == NULL) {
        return;
    }

    tinystories_free_bpe_heap_state(dataset->tokenizer);
    memset(dataset, 0, sizeof(*dataset));
}
