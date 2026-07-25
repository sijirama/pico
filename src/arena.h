/*
 *
 *

simply beautiful, multi-region, constant time ops, arena allocator, 
i'm like sooo fucking happy for this, no one has an idea

 * 
 * */

#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

#define MAX_ARENA_STACK 16
#define PICO_DEFAULT_ARENA_SIZE (32 * 1024 * 1024)

// INFO: arenas are for temporary pico objects. allocations are just pointer bumps,
// so there is no per-object free. reset the arena when the whole graph / scratch
// batch is dead, and every tensor allocated from it becomes invalid together.
struct ArenaBlock {
    struct ArenaBlock* next;
    size_t capacity;        // total size of the block, in bytes
    unsigned char* bottom;  // start of the malloc'd block
    unsigned char* curr;    // current position (the "offset" pointer)
};

struct Arena {
    struct ArenaBlock *begin, *end;
};

// The ctx stack is SHARED mutable state, so it must be ONE real global
// (extern here, defined once in global.c). It stays thread_local so each
// thread gets its own stack. (static-in-header would give every .c its own
// private copy, so ops.c couldn't see what main.c pushed.)
extern thread_local struct Arena* arena_stack[MAX_ARENA_STACK];
extern thread_local int arena_stack_top;

// INFO: one arena starts with one block. if the block fills up, we chain another
// block with the same capacity instead of reallocating and moving old pointers.
// that matters because tensors already handed out must keep their addresses.
static inline struct Arena* arena_init(size_t bytes) {
    struct Arena* arena = (struct Arena*)malloc(sizeof(struct Arena));
    if(arena == NULL) {
        return NULL;
    }

    struct ArenaBlock* block = (struct ArenaBlock*)malloc(sizeof(struct ArenaBlock));
    if(block == NULL) {
        return NULL;
    }

    block->bottom = (unsigned char*)malloc(bytes);
    if(block->bottom == NULL) {
        free(arena);  // don't leak the struct if the block alloc fails
        free(block);
        return NULL;
    }

    block->curr = block->bottom;  // we haven't used anything yet, so curr starts at bottom
    block->capacity = bytes;
    block->next = NULL;

    arena->begin = block;
    arena->end = arena->begin;  // begin and end will be the same at first

    return arena;
}

// INFO: this is the actual fast path. no metadata is stored per allocation, so
// there is nothing to free later except the whole block.
static inline void* arena_block_alloc(struct ArenaBlock* block, size_t size) {
    size_t used = block->curr - block->bottom;  // how much have we used so far?

    if(used + size > block->capacity) {  // bounds check: would this allocation overflow the block?
        return NULL;                     // out of memory, caller must check for this
    }

    void* ptr = block->curr;  // hand out the current position
    block->curr += size;      // move the pointer forward by `size` bytes
    return ptr;
}

// INFO: "realloc" here means grow the arena with another block, not realloc the
// existing block. moving an existing block would break every pointer we returned.
static inline void* arena_block_realloc(struct Arena* arena) {
    // create new block and make it arena->end

    size_t bytes = arena->begin->capacity;

    struct ArenaBlock* block = (struct ArenaBlock*)malloc(sizeof(struct ArenaBlock));
    if(block == NULL) {
        return NULL;
    }

    block->bottom = (unsigned char*)malloc(bytes);
    if(block->bottom == NULL) {
        free(arena);  // don't leak the struct if the block alloc fails
        free(block);
        return NULL;
    }

    block->curr = block->bottom;  // we haven't used anything yet, so curr starts at bottom
    block->capacity = bytes;
    block->next = NULL;

    arena->end->next = block;
    arena->end = block;

    return block;
}

// INFO: public allocation entry point. most callers should not call this directly;
// tensor/op constructors do it after resolving the arena from an explicit arg or
// the current ctx stack.
static inline void* arena_alloc(struct Arena* arena, size_t size) {
    void* ptr = arena_block_alloc(arena->end, size);
    if(ptr == NULL) {
        void* ptr2 = arena_block_realloc(arena);
        if(ptr2 == NULL) {
            printf("Memory allocation failed!\n");
            exit(1);
        }
        ptr = arena_block_alloc(arena->end, size);
    }
    return ptr;
}

static inline void arena_block_free(struct ArenaBlock* block) {
    free(block->bottom);  // free the actual data block (one real free)
    free(block);
}

// INFO: reset keeps the first block and drops every overflow block. this makes the
// common training loop cheap: build graph, backward, optimizer step, reset temp
// memory, then reuse the same first block on the next step.
static inline void arena_reset(struct Arena* arena) {
    struct ArenaBlock* current = arena->begin->next;  // start AFTER the first block
    struct ArenaBlock* nextBlock;

    while(current != NULL) {
        nextBlock = current->next;
        arena_block_free(current);
        current = nextBlock;
    }

    arena->begin->curr = arena->begin->bottom;
    arena->begin->next = NULL;
    arena->end = arena->begin;
}

// INFO: destroy is the real owner cleanup. after this, every pointer allocated
// from the arena is dead. persistent tensors are the exception because they never
// come from arena_alloc.
static inline void arena_destroy(struct Arena* arena) {
    // go through the entire list and delete each block

    struct ArenaBlock* current = arena->begin;
    struct ArenaBlock* nextBlock;

    while(current != NULL) {
        nextBlock = current->next;
        arena_block_free(current);
        current = nextBlock;
    }

    free(arena);
}

// ============================ arena context

// INFO: ctx is a small thread-local stack so callers can pass NULL into temp
// allocation APIs. explicit arena args still win, but NULL means "use current".
static inline void arena_ctx_push(struct Arena* arena) {
    arena_stack_top++;
    arena_stack[arena_stack_top] = arena;
}

// NOTE: push/pop are intentionally tiny right now. callers have to keep them
// balanced; later we can add debug checks if this starts biting us.
static inline void arena_ctx_pop(void) {
    arena_stack_top--;
}

static inline struct Arena* arena_ctx_current(void) {
    if(arena_stack_top == -1 || arena_stack_top == MAX_ARENA_STACK) {
        return NULL;
    }
    return arena_stack[arena_stack_top];
}

// INFO: this is the convention point. every temporary allocation API should call
// this first: use the arena passed by the caller, or fall back to the ctx arena.
static inline struct Arena* arena_resolve(struct Arena* arena) {
    if(arena != NULL) {
        return arena;
    }
    return arena_ctx_current();
}
