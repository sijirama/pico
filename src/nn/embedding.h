#pragma once

#include <stddef.h>

#include "arena.h"
#include "ctx.h"
#include "tensor.h"

struct PicoEmbedding {
    struct PicoTensor* table;  // [num_embeddings, embedding_dim]
    int num_embeddings;        // size of the vocabulary (how many distinct tokens/words you have)
    int embedding_dim;         // size of the vector space each token is embedded into
};

static inline void pico_embedding_backward(struct PicoTensor* self) {
    struct PicoTensor* table = self->parents[0];
    struct PicoTensor* input_indices = self->parents[1];

    int embedding_dim = table->shape[1];
    int64_t seq_len = input_indices->shape[0];

    for(int64_t i = 0; i < seq_len; i++) {
        int64_t idx = (int64_t)input_indices->data[i];

        float* grad_out_row = (float*)self->grad + i * embedding_dim;
        float* grad_table_row = (float*)table->grad + idx * embedding_dim;

        for(int j = 0; j < embedding_dim; j++) {
            grad_table_row[j] += grad_out_row[j];  // += is the whole point
        }
    }
}

/*
  input indeices is a 1d vector, basically an array - [1,2,3,4]
  our embedding vector is a lookup table to get the embed vector for each indices
  so our return is gonna be the input indices, but for each we've gotten our embedding vector - a bettrer representation
  [input_indices x embed_vector/embed_dim]
  */
static inline struct PicoTensor* pico_embedding_apply(struct PicoContext* ctx, struct PicoEmbedding* embedding,
                                                     struct PicoTensor* input_indices) {
    if(ctx == NULL || embedding == NULL || embedding->table == NULL || input_indices == NULL) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for embedding allocation\n");
        return NULL;
    }

    // Assert that input_indices is one dimensional
    if(input_indices->ndim != 1) {
        fprintf(stderr, "PicoEmbeddingError: input_indices must be 1-dimensional\n");
        return NULL;
    }

    int ndim = 2;
    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    res_shape[0] = input_indices->shape[0];
    res_shape[1] = embedding->embedding_dim;

    struct PicoTensor* out = pico_create_tensor(ctx, res_shape, ndim);

    // For each index in input_indices, copy the corresponding embedding vector from embedding->table
    for(int64_t i = 0; i < input_indices->shape[0]; i++) {
        int64_t idx = (int64_t)input_indices->data[i];  // for each index in the input_indices
        if(idx < 0 || idx >= embedding->num_embeddings) {
            fprintf(stderr, "PicoEmbeddingError: index %ld out of range\n", idx);
            return NULL;
        }
        float* src =
            (float*)embedding->table->data + idx * embedding->embedding_dim;  // get the row of the index in table
        float* dst = (float*)out->data + i * embedding->embedding_dim;        // get the row of index in the out tensor
        for(int j = 0; j < embedding->embedding_dim; j++) {                   // drop every embed dim from src to dest
            dst[j] = src[j];
        }
    }

    // Set the parent tensor for autograd
    out->num_parents = 2;
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = embedding->table;
    out->parents[1] = input_indices;
    out->_backward = pico_embedding_backward;
    out->backend = input_indices->backend;

    return out;
}

static inline struct PicoEmbedding* pico_embedding_init(struct PicoContext* ctx, int num_embeddings, int embedding_dim) {
    if(ctx == NULL || num_embeddings <= 0 || embedding_dim <= 0) {
        return NULL;
    }

    //
    //
    // num_embeddings: is the size of the vocabulary (how many distinct tokens/words you have).
    // embedding_dim: is the size of the vector space each token is embedded into.
    // input_indices: are indices of tokens, and the output will be a tensor containing their corresponding embeddings.

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for embedding allocation\n");
        return NULL;
    }

    struct PicoEmbedding* embedding = arena_alloc(arena, sizeof(struct PicoEmbedding));
    if(embedding == NULL) {
        return NULL;
    }

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * 2);
    if(res_shape == NULL) {
        return NULL;
    }

    res_shape[0] = num_embeddings;
    res_shape[1] = embedding_dim;

    struct PicoTensor* table = pico_param(ctx, res_shape, 2);
    if(table == NULL) {
        return NULL;
    }

    embedding->table = table;
    embedding->num_embeddings = num_embeddings;
    embedding->embedding_dim = embedding_dim;

    return embedding;
}
