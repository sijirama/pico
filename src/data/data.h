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

// INFO: this is the dataset contract. any dataset type can work with pico as
// long as it can tell us its length, return one item by index, and clean up any
// private state it owns.
struct DatasetVTable {
    size_t (*len)(const struct Dataset* self);
    struct DatasetItem (*get)(const struct Dataset* self, size_t idx);
    void (*free)(struct Dataset* self);
};

// INFO: funcs is the public interface, data is the private body. for a text
// dataset this can point to TextDatasetData, for csv it can point to
// CsvDatasetData, for images it can point to ImageDatasetData. pico does not
// care about the real type, only the dataset functions do.
struct Dataset {
    const struct DatasetVTable* funcs;
    void* data;
};

// INFO: dataloader does not know if the data came from text, csv, images, or
// a remote source. it only asks the dataset for items and groups them together
// into batches.
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
