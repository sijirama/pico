/*
 * ============================================================================
 *  AUTOGRAD — per-op backward functions
 * ============================================================================
 *
 *  _backward(self) is called on ONE tensor (the output of an op). Its job:
 *  take the gradient that has flowed INTO self, and push it back into self's
 *  parents (accumulate into each parent's ->grad).
 *
 *  THE TWO DERIVATIVES (this is the whole idea):
 *    self->grad  = dL/dself   GLOBAL  — loss w.r.t. self; the "upstream" gradient,
 *                                       already filled by self's consumers (or
 *                                       seeded to 1 if self is the output/loss).
 *    d(self)/da  = LOCAL       — how self depends on a, for THIS op alone
 *                                (its own wiggle-factor). never stored; just a
 *                                multiplier we compute from the op's formula.
 *
 *  Every tensor's ->grad means dL/d(that tensor), all w.r.t. the ONE loss L.
 *
 *  THE CHAIN RULE, in code:
 *      a->grad[i]  +=   self->grad[i]   *   d(self)/da
 *         │                │                    │
 *       dL/da          dL/dself (global)    local factor
 *
 *  TWO DIFFERENT OPERATIONS (don't blur them):
 *    '*' MULTIPLY  = chain rule (global × local) — extends the chain ONE step back.
 *    '+=' ACCUMULATE = SUM the contributions a tensor gets from MULTIPLE paths
 *         (used in multiple ops, or both parents being the same tensor).
 *  '+=' (not '=') because those path-contributions must SUM. (proof: y=a+a -> dy/da=2.)
 *
 *  LOCAL DERIVATIVES PER OP (g = self->grad):
 *    add   self=a+b   d/da=1,  d/db=1     -> a.grad += g;      b.grad += g
 *    sub   self=a-b   d/da=1,  d/db=-1    -> a.grad += g;      b.grad -= g
 *    mul   self=a*b   d/da=b,  d/db=a     -> a.grad += g*b;    b.grad += g*a
 *    relu  self=max(0,x)  d/dx = (x>0?1:0) -> x.grad += g*(x>0?1:0)
 *          a GATE: passes grad where input was +, blocks it where -. an off neuron
 *          gets 0 grad; one stuck negative never updates = "dying ReLU".
 *    softmax  every output depends on ALL inputs (shared denom) -> not element-wise,
 *             it's a Jacobian:  dz_j += s_j * (g_j - dot(g, s))   // dot = Σ_i g_i*s_i
 *    matmul  C=A·B   ->  dA = dC·Bᵀ ,  dB = Aᵀ·dC   (matrix, not element-wise)
 *  (rule: element-wise op = each output sees ONE input; softmax = each sees ALL.)
 *
 * ============================================================================
 */

#pragma once

#include <stdint.h>
#include "tensor.h"

static inline void pico_add_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];
    struct PicoTensor* b = self->parents[1];

    int64_t ia = 0;
    int64_t ib = 0;

    for(int64_t i = 0; i < self->numel; i++) {
        ia = map_index(i, a, self->strides, self->ndim);
        ib = map_index(i, b, self->strides, self->ndim);
        a->grad[ia] += self->grad[i];
        b->grad[ib] += self->grad[i];
    }
}

static inline void pico_sub_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];
    struct PicoTensor* b = self->parents[1];

    int64_t ia = 0;
    int64_t ib = 0;
    for(int64_t i = 0; i < self->numel; i++) {
        ia = map_index(i, a, self->strides, self->ndim);
        ib = map_index(i, b, self->strides, self->ndim);
        a->grad[ia] += self->grad[i];
        b->grad[ib] -= self->grad[i];
    }
}

static inline void pico_mul_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];
    struct PicoTensor* b = self->parents[1];

    int64_t ia = 0;
    int64_t ib = 0;
    for(int64_t i = 0; i < self->numel; i++) {
        ia = map_index(i, a, self->strides, self->ndim);
        ib = map_index(i, b, self->strides, self->ndim);
        a->grad[ia] += self->grad[i] * b->data[ib];
        b->grad[ib] += self->grad[i] * a->data[ia];
    }
}

// C = A·B   ->   dA = dC·Bᵀ ,  dB = Aᵀ·dC   (dC = self->grad)
// two matmul-style triple loops; transpose is baked into the index order.
static inline void pico_matmul_backward(struct PicoTensor* self) {
    struct PicoTensor* a = self->parents[0];  // A (M,K)
    struct PicoTensor* b = self->parents[1];  // B (K,N)

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    // dA[i][k] = Σ_j dC[i][j] * B[k][j]
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            float acc = 0.0f;
            for(int j = 0; j < N; j++) {
                acc += self->grad[i * self->strides[0] + j * self->strides[1]] *
                       b->data[k * b->strides[0] + j * b->strides[1]];
            }
            a->grad[i * a->strides[0] + k * a->strides[1]] += acc;  // += : accumulate across consumers
        }
    }

    // dB[k][j] = Σ_i A[i][k] * dC[i][j]
    for(int k = 0; k < K; k++) {
        for(int j = 0; j < N; j++) {
            float acc = 0.0f;
            for(int i = 0; i < M; i++) {
                acc += a->data[i * a->strides[0] + k * a->strides[1]] *
                       self->grad[i * self->strides[0] + j * self->strides[1]];
            }
            b->grad[k * b->strides[0] + j * b->strides[1]] += acc;
        }
    }
}
