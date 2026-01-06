#pragma once

typedef enum {
    NONE, // for leaf tensors 
    ADD,
    SUB,
    MUL,
    DIV
} GradientOp;
