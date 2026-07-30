#pragma once

#include <stddef.h>

struct PicoContext;
struct Tokenizer;

// INFO: tokenizer contract. wordbased, wordpiece, bpe, or any other tokenizer
// can plug in here as long as it can report vocab length, encode text to ids,
// and decode ids back to text.
struct TokenizerVTable {
    size_t (*len)(const struct Tokenizer* self);
    void* (*encode)(const struct Tokenizer* self, const char* text);
    void* (*decode)(const struct Tokenizer* self, const size_t* ids);
};

// INFO: methods is the public interface, data is the tokenizer-specific body.
// for wordbased this points to WordBasedPicoTKData, but another tokenizer can
// use a completely different struct without changing the caller.
struct Tokenizer {
    const struct TokenizerVTable* methods;
    struct PicoContext* ctx;
    void* data;
};
