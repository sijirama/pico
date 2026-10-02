#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "pico.h"

#define TINYSTORIES_DEFAULT_TRAIN_PATH "datasets/tinystories/train_5mb.txt"
#define TINYSTORIES_DEFAULT_VALID_PATH "datasets/tinystories/valid_1mb.txt"

struct TinyStoriesDatasetConfig {
    const char* path;
    size_t max_rows;
    size_t batch_size;
    bool shuffle;
    int max_vocab_size;
    int max_seq_len;
};

struct TinyStoriesDataset {
    struct PicoContext* ctx;
    struct Tokenizer* tokenizer;
    struct Dataset dataset;
    struct DataLoader* loader;
    char** texts;
    size_t len;
    int max_seq_len;
};

struct TinyStoriesDatasetConfig tinystories_default_config(void);
bool tinystories_dataset_prepare(struct PicoContext* ctx, struct TinyStoriesDataset* out,
                                 struct TinyStoriesDatasetConfig config);
void tinystories_dataset_reset(struct TinyStoriesDataset* dataset);
void tinystories_dataset_free(struct TinyStoriesDataset* dataset);
