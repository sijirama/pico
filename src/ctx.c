#include "ctx.h"

#include "tensor.h"

// INFO: context owns two arenas for one training/runtime session. temp_arena is
// for graph outputs/intermediates, param_arena is for trainable params.
struct PicoContext pico_context_init(void) {
    struct PicoContext ctx;
    ctx.temp_arena = arena_init(PICO_DEFAULT_ARENA_SIZE);
    ctx.param_arena = arena_init(PICO_DEFAULT_ARENA_SIZE);
    ctx.arena = ctx.temp_arena;
    ctx.mode = PICO_TRAIN;
    pico_vec_init(&ctx.params, 16);
    return ctx;
}

struct Arena* pico_context_temp_arena(struct PicoContext* ctx) {
    if(ctx == NULL || ctx->temp_arena == NULL) {
        return NULL;
    }
    return ctx->temp_arena;
}

struct Arena* pico_context_param_arena(struct PicoContext* ctx) {
    if(ctx == NULL || ctx->param_arena == NULL) {
        return NULL;
    }
    return ctx->param_arena;
}

struct Arena* pico_context_arena(struct PicoContext* ctx) {
    return pico_context_temp_arena(ctx);
}

void pico_context_register_param(struct PicoContext* ctx, struct PicoTensor* param) {
    if(ctx == NULL || param == NULL) {
        return;
    }
    pico_vec_push(&ctx->params, param);
}

// INFO: destroy mirrors init. temp tensors die with temp_arena; params die with
// param_arena after serializers/optimizers are done with ctx->params.
void pico_context_destroy(struct PicoContext* ctx) {
    if(ctx == NULL) {
        return;
    }

    pico_vec_free(&ctx->params);

    if(ctx->temp_arena != NULL) {
        arena_destroy(ctx->temp_arena);
        ctx->temp_arena = NULL;
        ctx->arena = NULL;
    }

    if(ctx->param_arena != NULL) {
        arena_destroy(ctx->param_arena);
        ctx->param_arena = NULL;
    }

    if(ctx->arena != NULL) {
        ctx->arena = NULL;
    }
}
