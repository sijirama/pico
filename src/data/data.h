/*

#INFO:

Dataset stores samples, DataLoader turns that dataset into batches.

The dataset side is intentionally user-defined: the vtable tells pico how to ask
for length and items, while `data` points to whatever backing storage the user
needs, like csv rows, token ids, images, or generated samples.

 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "ctx.h"

struct PicoTensor;
struct Dataset;

struct DatasetItem {
    struct PicoTensor* x;
    struct PicoTensor* y;
};

struct DataBatch {
    struct DatasetItem* items;
    size_t size;
};

struct DatasetVTable {
    size_t (*len)(const struct Dataset* self);
    struct DatasetItem (*get)(const struct Dataset* self, size_t idx);
    void (*free)(struct Dataset* self);
};

struct Dataset {
    const struct DatasetVTable* funcs;
    void* data;
};

struct DataLoader {
    struct PicoContext* ctx;
    struct Dataset* dataset;
    size_t batch_size;
    bool shuffle;
    size_t cursor;
    size_t* indices;
};

struct DataLoader* pico_dataloader_init(struct PicoContext* ctx, struct Dataset* dataset, size_t batch_size,
                                        bool shuffle);
struct DataBatch* pico_dataloader_next(struct DataLoader* loader);
void pico_dataloader_reset(struct DataLoader* loader);
