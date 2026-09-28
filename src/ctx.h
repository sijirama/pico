// INFO: pico context is the state for one training/runtime session.
#pragma once

#include "arena.h"
#include "lib/pico_vector.h"

struct PicoTensor;

enum PicoMode { PICO_TRAIN, PICO_EVAL };

struct PicoContext {
    struct Arena *temp_arena;
    struct Arena *param_arena;
    // TODO: compatibility alias while older code is migrated to temp_arena.
    struct Arena *arena;
    struct PicoVec params; // list of persistent tensors created through pico_param
    enum PicoMode mode;
};

struct PicoContext pico_context_init(void);
struct Arena *pico_context_temp_arena(struct PicoContext *ctx);
struct Arena *pico_context_param_arena(struct PicoContext *ctx);
struct Arena *pico_context_arena(struct PicoContext *ctx);
void pico_context_register_param(struct PicoContext *ctx, struct PicoTensor *param);
void pico_context_destroy(struct PicoContext *ctx);
