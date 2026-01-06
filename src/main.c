#include <stdio.h>
#include <stdlib.h>

#include "autograd.h"
#include "tensor.h"

int main() {
    printf("=== AUTOGRAD TEST ===\n\n");

    // Create input and weights
    struct Tensor* x = tensor_from_data((float[]){2.0}, (int[]){1}, 1);
    struct Tensor* w = tensor_from_data((float[]){3.0}, (int[]){1}, 1);

    // Mark weight as trainable
    w->requires_grad = 1;
    w->grad = malloc(w->numel * sizeof(float));
    for(int i = 0; i < w->numel; i++)
        w->grad[i] = 0.0f;

    printf("Initial values:\n");
    printf("x = %.1f\n", x->data[0]);
    printf("w = %.1f (requires_grad=1)\n\n", w->data[0]);

    // Forward pass: y = w * x
    struct Tensor* y = tensor_dot(w, x);
    printf("Forward pass: y = w * x = %.1f\n\n", y->data[0]);

    // Backward pass
    printf("Calling y.backward()...\n");
    float* grad_output = malloc(y->numel * sizeof(float));
    grad_output[0] = 1.0;  // Start with gradient of 1.0

    tensor_backward(y, grad_output);

    printf("\nAfter backward:\n");
    printf("w.grad = %.1f\n", w->grad[0]);
    printf("Expected: dy/dw = x = %.1f\n\n", x->data[0]);

    // Cleanup
    free(grad_output);
    tensor_free(x);
    tensor_free(w);
    tensor_free(y);

    printf("\n=== TEST 2: CHAIN RULE ===\n\n");

    // Test chain: loss = (w * x)^2
    struct Tensor* x2 = tensor_from_data((float[]){2.0}, (int[]){1}, 1);
    struct Tensor* w2 = tensor_from_data((float[]){3.0}, (int[]){1}, 1);

    w2->requires_grad = 1;
    w2->grad = malloc(w2->numel * sizeof(float));
    w2->grad[0] = 0.0f;

    printf("x = %.1f, w = %.1f\n", x2->data[0], w2->data[0]);

    struct Tensor* a = tensor_dot(w2, x2);   // a = 6.0
    struct Tensor* loss = tensor_dot(a, a);  // loss = 36.0

    printf("a = w * x = %.1f\n", a->data[0]);
    printf("loss = a * a = %.1f\n\n", loss->data[0]);

    float* loss_grad = malloc(loss->numel * sizeof(float));
    loss_grad[0] = 1.0;

    tensor_backward(loss, loss_grad);

    printf("After backward:\n");
    printf("w2.grad = %.1f\n", w2->grad[0]);
    printf("Expected: dLoss/dw = 2*a*x = 2*6*2 = %.1f\n", 2 * a->data[0] * x2->data[0]);

    free(loss_grad);
    tensor_free(x2);
    tensor_free(w2);
    tensor_free(a);
    tensor_free(loss);

    return 0;
}
