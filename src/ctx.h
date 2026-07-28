// INFO: pico context is the state for one training/runtime session.
#pragma once

#include "arena.h"
#include "lib/pico_vector.h"

enum PicoMode { PICO_TRAIN, PICO_EVAL };

struct PicoContext {
    struct Arena* arena;
    enum PicoMode mode;
    struct PicoVec params;   // list of persistent tensors created through pico_param
};

struct PicoContext pico_context_init(void);
struct Arena* pico_context_arena(struct PicoContext* ctx);
void pico_context_register_param(struct PicoContext* ctx, struct PicoTensor* param);
void pico_context_destroy(struct PicoContext* ctx);
