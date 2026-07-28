/*

 INFO: PicoContext should be constructed as the state of one training/runtime session.

 */
#pragma once

#include "arena.h"
#include "lib/pico_vector.h"

enum PicoMode { PICO_TRAIN, PICO_EVAL };

struct PicoContext {
    struct Arena* arena;
    enum PicoMode mode;
    struct PicoVec params;   // list of trainable tensors
};

// INFO: context owns the default arena for one training/runtime session. params
// are still heap-backed tensors, but temps/intermediates should come from here.
static inline struct PicoContext pico_context_init(void) {
    struct PicoContext ctx;
    ctx.arena = arena_init(PICO_DEFAULT_ARENA_SIZE);
    ctx.mode = PICO_TRAIN;
    pico_vec_init(&ctx.params, 16);
    return ctx;
}

// INFO: destroy mirrors init. this frees the param registry storage and the
// context-owned arena; it does not free the trainable tensors inside params.
static inline void pico_context_destroy(struct PicoContext* ctx) {
    if(ctx == NULL) {
        return;
    }

    pico_vec_free(&ctx->params);

    if(ctx->arena != NULL) {
        arena_destroy(ctx->arena);
        ctx->arena = NULL;
    }
}
