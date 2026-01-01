#include <stdio.h>

#include "tensor.h"

int main() {
    struct Tensor* x = tensor_from_data((float[]){1.1, 2.2, 3.3}, (int[]){3}, 1);
    struct Tensor* y = tensor_from_data((float[]){1.1, 2.2, 3.3}, (int[]){3}, 1);

    tensor_add(x, y);
    tensor_matmul(x, y);

    printf("Resulting Tensor Data:\n");
    for(int i = 0; i < x->numel; i++) {
        printf("%f ", x->data[i]);
    }

    printf("\n");

    printf("Sum of tensor x is %f \n", tensor_sum(x));
    printf("Mean of tensor x is %f \n", tensor_mean(x));
    printf("Max of tensor x is %f \n", tensor_max(x));

    tensor_free(x);
    tensor_free(y);

    return 0;
}
