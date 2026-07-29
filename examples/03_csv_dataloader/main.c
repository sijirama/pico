#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <csv.h>

#include "pico.h"

struct CsvDatasetData {
    struct PicoContext* ctx;
    float* xs;
    float* ys;
    size_t len;
    size_t capacity;
};

struct CsvParseState {
    struct CsvDatasetData* dataset;
    float fields[3];
    size_t field_idx;
    bool skip_header;
};

static size_t csv_dataset_len(const struct Dataset* dataset) {
    struct CsvDatasetData* data = (struct CsvDatasetData*)dataset->data;
    return data->len;
}

static struct DatasetItem csv_dataset_get(const struct Dataset* dataset, size_t idx) {
    struct CsvDatasetData* data = (struct CsvDatasetData*)dataset->data;
    int64_t x_shape[] = {1, 2};
    int64_t y_shape[] = {1, 1};

    struct DatasetItem item = {
        .x = pico_tensor_from_data(data->ctx, x_shape, 2, &data->xs[idx * 2]),
        .y = pico_tensor_from_data(data->ctx, y_shape, 2, &data->ys[idx]),
    };
    return item;
}

static void csv_dataset_free(struct Dataset* dataset) {
    if(dataset == NULL || dataset->data == NULL) {
        return;
    }

    struct CsvDatasetData* data = (struct CsvDatasetData*)dataset->data;
    free(data->xs);
    free(data->ys);
    free(data);
    dataset->data = NULL;
}

static const struct DatasetVTable CSV_DATASET_FUNCS = {
    .len = csv_dataset_len,
    .get = csv_dataset_get,
    .free = csv_dataset_free,
};

static bool csv_dataset_reserve(struct CsvDatasetData* dataset, size_t wanted) {
    if(wanted <= dataset->capacity) {
        return true;
    }

    size_t next_capacity = dataset->capacity == 0 ? 8 : dataset->capacity * 2;
    while(next_capacity < wanted) {
        next_capacity *= 2;
    }

    float* xs = malloc(sizeof(float) * next_capacity * 2);
    if(xs == NULL) {
        return false;
    }

    float* ys = malloc(sizeof(float) * next_capacity);
    if(ys == NULL) {
        free(xs);
        return false;
    }

    if(dataset->len > 0) {
        memcpy(xs, dataset->xs, sizeof(float) * dataset->len * 2);
        memcpy(ys, dataset->ys, sizeof(float) * dataset->len);
    }

    free(dataset->xs);
    free(dataset->ys);

    dataset->xs = xs;
    dataset->ys = ys;
    dataset->capacity = next_capacity;
    return true;
}

static void csv_field_cb(void* field, size_t len, void* user_data) {
    struct CsvParseState* state = (struct CsvParseState*)user_data;
    if(state->skip_header) {
        return;
    }

    if(state->field_idx >= 3) {
        return;
    }

    char value[32];
    size_t copy_len = len < sizeof(value) - 1 ? len : sizeof(value) - 1;
    memcpy(value, field, copy_len);
    value[copy_len] = '\0';

    state->fields[state->field_idx] = strtof(value, NULL);
    state->field_idx += 1;
}

static void csv_row_cb(int c, void* user_data) {
    (void)c;

    struct CsvParseState* state = (struct CsvParseState*)user_data;
    if(state->skip_header) {
        state->skip_header = false;
        state->field_idx = 0;
        return;
    }

    if(state->field_idx != 3) {
        state->field_idx = 0;
        return;
    }

    size_t row = state->dataset->len;
    if(!csv_dataset_reserve(state->dataset, row + 1)) {
        state->field_idx = 0;
        return;
    }

    state->dataset->xs[row * 2 + 0] = state->fields[0];
    state->dataset->xs[row * 2 + 1] = state->fields[1];
    state->dataset->ys[row] = state->fields[2];
    state->dataset->len += 1;
    state->field_idx = 0;
}

static struct Dataset csv_dataset_from_file(struct PicoContext* ctx, const char* path) {
    struct Dataset dataset = {0};
    FILE* file = fopen(path, "rb");
    if(file == NULL) {
        return dataset;
    }

    struct CsvDatasetData* data = calloc(1, sizeof(struct CsvDatasetData));
    if(data == NULL) {
        fclose(file);
        return dataset;
    }
    data->ctx = ctx;

    struct CsvParseState state = {
        .dataset = data,
        .field_idx = 0,
        .skip_header = true,
    };

    struct csv_parser parser;
    if(csv_init(&parser, CSV_STRICT) != 0) {
        free(data);
        fclose(file);
        return dataset;
    }

    char buffer[1024];
    size_t bytes_read = 0;
    while((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        csv_parse(&parser, buffer, bytes_read, csv_field_cb, csv_row_cb, &state);
    }
    csv_fini(&parser, csv_field_cb, csv_row_cb, &state);
    csv_free(&parser);
    fclose(file);

    dataset.funcs = &CSV_DATASET_FUNCS;
    dataset.data = data;
    return dataset;
}

int main(void) {
    pico_init();
    struct PicoContext ctx = pico_context_init();

    struct Dataset dataset = csv_dataset_from_file(&ctx, "examples/03_csv_dataloader/data.csv");
    if(dataset.funcs == NULL) {
        fprintf(stderr, "failed to load csv dataset\n");
        pico_context_destroy(&ctx);
        pico_shutdown();
        return 1;
    }

    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);
    struct DataBatch* batch = NULL;
    while((batch = pico_dataloader_next(loader)) != NULL) {
        printf("batch size: %zu\n", batch->size);
        for(size_t i = 0; i < batch->size; i++) {
            pico_tensor_print(batch->items[i].x);
            pico_tensor_print(batch->items[i].y);
        }
    }

    dataset.funcs->free(&dataset);
    pico_context_destroy(&ctx);
    pico_shutdown();
    return 0;
}
