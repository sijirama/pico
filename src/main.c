#include "tensor.h"

int main() {
    struct Tensor my_tensor;

    my_tensor.data = (float[]){1.0, 2.0, 3.0, 4.0};

    tensor_to_cpu(&my_tensor);

    my_tensor.ops->matmul(&my_tensor, &my_tensor);

    tensor_to_gpu(&my_tensor);

    my_tensor.ops->matmul(&my_tensor, &my_tensor);

    return 0;
}
