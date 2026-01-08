
#pragma once
#include <stddef.h>

#include "tensor.h"

struct ContextGraphOps;

// NOTE:when you implement append, remember that if size == capacity, you’ll need to realloc the
// registry. A common trick is to double the capacity each time (capacity *= 2) to keep the number
// of allocations low.

struct Graph {
    struct ContextGraphOps* ops;
    Tensor** registry;
    size_t capacity;  // max capacity
    size_t size;      // current graph size
};

struct ContextGraphOps {
    void (*append)(struct Graph* graph, struct Tensor* x);
    void (*reset)(struct Graph* graph);
};

struct Graph* graph_init();
