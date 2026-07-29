#pragma once

#include <stddef.h>

struct Tokenizer;

struct TokenizerVTable {
    size_t (*len)(const struct Tokenizer* self);
    void* (*encode)(const struct Tokenizer* self, const char* text);
    void* (*decode)(const struct Tokenizer* self, const float* ids);
};

struct Tokenizer {
    const struct TokenizerVTable* methods;
    struct PicoContext* ctx;
    void * data;
};
