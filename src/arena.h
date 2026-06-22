/*
 *
 *

simply beautiful, multiregion, constant time, arena allocator, 
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

static inline void* arena_block_alloc(struct ArenaBlock* block, size_t size) {
    size_t used = block->curr - block->bottom;  // how much have we used so far?

    if(used + size > block->capacity) {  // bounds check: would this allocation overflow the block?
        return NULL;                     // out of memory, caller must check for this
    }

    void* ptr = block->curr;  // hand out the current position
    block->curr += size;      // move the pointer forward by `size` bytes
    return ptr;
}

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

static inline void arena_ctx_push(struct Arena* arena) {
    arena_stack_top++;
    arena_stack[arena_stack_top] = arena;
}

static inline void arena_ctx_pop(void) {
    arena_stack_top--;
}

static inline struct Arena* arena_ctx_current(void) {
    if(arena_stack_top == -1 || arena_stack_top == MAX_ARENA_STACK) {
        return NULL;
    }
    return arena_stack[arena_stack_top];
}
