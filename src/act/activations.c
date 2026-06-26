
#include "activations.h"

#include "arena.h"
#include "autograd.h"
#include "tensor.h"

struct PicoTensor* pico_relu(struct PicoTensor* x) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }
    struct PicoTensor* out = pico_create_tensor(arena, x->shape, x->ndim);

    for(int i = 0; i < x->numel; i++) {
        out->data[i] = MAX(x->data[i], 0);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = x;
    out->num_parents = 1;
    out->_backward = pico_relu_backward;

    return out;
}
