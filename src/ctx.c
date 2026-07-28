#include "ctx.h"

#include "tensor.h"

// INFO: context owns the default arena for one training/runtime session. params
// are heap-backed, while temps/intermediates come from the ctx arena.
struct PicoContext pico_context_init(void) {
    struct PicoContext ctx;
    ctx.arena = arena_init(PICO_DEFAULT_ARENA_SIZE);
    ctx.mode = PICO_TRAIN;
    pico_vec_init(&ctx.params, 16);
    return ctx;
}

struct Arena* pico_context_arena(struct PicoContext* ctx) {
    if(ctx == NULL || ctx->arena == NULL) {
        return NULL;
    }
    return ctx->arena;
}

void pico_context_register_param(struct PicoContext* ctx, struct PicoTensor* param) {
    if(ctx == NULL || param == NULL) {
        return;
    }
    pico_vec_push(&ctx->params, param);
}

// INFO: destroy mirrors init. heap params are freed one by one, and temp tensors
// die together when the arena is destroyed.
void pico_context_destroy(struct PicoContext* ctx) {
    if(ctx == NULL) {
        return;
    }

    while(ctx->params.size > 0) {
        struct PicoTensor* param = ctx->params.data[ctx->params.size - 1];
        ctx->params.size--;
        pico_tensor_free_heap(param);
    }

    pico_vec_free(&ctx->params);

    if(ctx->arena != NULL) {
        arena_destroy(ctx->arena);
        ctx->arena = NULL;
    }
}
