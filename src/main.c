#include <stdio.h>

#include "tensor.h"

int main() {
    struct Tensor* x = tensor_from_data((float[]){1.1, 2.2, 3.3}, (int[]){3}, 1);
    struct Tensor* y = tensor_from_data((float[]){1.1, 2.2, 3.3}, (int[]){3}, 1);

    tensor_add(x, y);

    printf("Resulting Tensor Data:\n");
    for(int i = 0; i < x->numel; i++) {
        printf("%f ", x->data[i]);
    }
    printf("\n");

    tensor_free(x);
    tensor_free(y);

    return 0;
}
