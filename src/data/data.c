/*

#INFO:

Dataset: stores the samples and their corresponding labels
DataLoader: wraps an iterable around the Dataset to enable easy access to the samples.

 */

#include <stdlib.h>

#include "data/data.h"

static void shuffle_indices(size_t* indices, size_t n) {
    if(n < 2) {
        return;
    }

    for(size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

struct DataLoader* pico_dataloader_init(struct PicoContext* ctx, struct Dataset* dataset, size_t batch_size,
                                        bool shuffle) {
    if(ctx == NULL || ctx->arena == NULL || dataset == NULL || dataset->funcs == NULL || dataset->funcs->len == NULL ||
       batch_size == 0) {
        return NULL;
    }

    struct DataLoader* loader = arena_alloc(ctx->arena, sizeof(struct DataLoader));
    if(loader == NULL) {
        return NULL;
    }

    size_t n = dataset->funcs->len(dataset);

    loader->ctx = ctx;
    loader->dataset = dataset;
    loader->cursor = 0;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->indices = arena_alloc(ctx->arena, sizeof(size_t) * n);

    if(loader->indices == NULL) {
        return NULL;
    }

    for(size_t i = 0; i < n; i++) {
        loader->indices[i] = i;
    }

    if(shuffle) {
        shuffle_indices(loader->indices, n);
    }

    return loader;
}

// INFO: next is just cursor movement over the index list. the dataset still owns
// how a sample is created, the loader only decides which indices belong together.
struct DataBatch* pico_dataloader_next(struct DataLoader* loader) {
    if(loader == NULL || loader->ctx == NULL || loader->ctx->arena == NULL || loader->dataset == NULL ||
       loader->dataset->funcs == NULL || loader->dataset->funcs->len == NULL ||
       loader->dataset->funcs->get == NULL || loader->indices == NULL) {
        return NULL;
    }

    size_t n = loader->dataset->funcs->len(loader->dataset);
    if(loader->cursor >= n) {
        return NULL;
    }

    size_t remaining = n - loader->cursor;
    size_t batch_size = loader->batch_size < remaining ? loader->batch_size : remaining;

    struct DataBatch* batch = arena_alloc(loader->ctx->arena, sizeof(struct DataBatch));
    if(batch == NULL) {
        return NULL;
    }

    batch->items = arena_alloc(loader->ctx->arena, sizeof(struct DatasetItem) * batch_size);
    if(batch->items == NULL) {
        return NULL;
    }
    batch->size = batch_size;

    for(size_t i = 0; i < batch_size; i++) {
        size_t dataset_idx = loader->indices[loader->cursor + i];
        batch->items[i] = loader->dataset->funcs->get(loader->dataset, dataset_idx);
    }

    loader->cursor += batch_size;
    return batch;
}

// INFO: reset starts a new epoch. when shuffle is on, the same indices are reused
// but reordered before the next call to pico_dataloader_next.
void pico_dataloader_reset(struct DataLoader* loader) {
    if(loader == NULL || loader->dataset == NULL || loader->dataset->funcs == NULL ||
       loader->dataset->funcs->len == NULL || loader->indices == NULL) {
        return;
    }

    size_t n = loader->dataset->funcs->len(loader->dataset);
    loader->cursor = 0;

    for(size_t i = 0; i < n; i++) {
        loader->indices[i] = i;
    }

    if(loader->shuffle) {
        shuffle_indices(loader->indices, n);
    }
}
