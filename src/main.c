#include "tensor.h"

int main() {
    // --- TEST 1: Identical Shapes [3] + [3] ---
    struct Tensor* a1 = tensor_from_data((float[]){1, 2, 3}, (int[]){3}, 1);
    struct Tensor* b1 = tensor_from_data((float[]){10, 20, 30}, (int[]){3}, 1);
    tensor_add(a1, b1);
    print_tensor("Test 1 Result", a1);  // Expected: 11, 22, 33

    // --- TEST 2: Scalar Broadcasting [3] + [1] ---
    // Adding 10.0 to every element in [1, 2, 3]
    struct Tensor* a2 = tensor_from_data((float[]){1, 2, 3}, (int[]){3}, 1);
    struct Tensor* b2 = tensor_from_data((float[]){10.0}, (int[]){1}, 1);
    tensor_add(a2, b2);
    print_tensor("Test 2 Result", a2);  // Expected: 11, 12, 13

    // --- TEST 3: Row Broadcasting [2, 3] + [1, 3] ---
    // Matrix: [[1, 2, 3], [4, 5, 6]]
    // Row:    [[10, 20, 30]]
    struct Tensor* a3 = tensor_from_data((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);
    struct Tensor* b3 = tensor_from_data((float[]){10, 20, 30}, (int[]){1, 3}, 2);
    tensor_add(a3, b3);
    print_tensor("Test 3 Result", a3);  // Expected: 11, 22, 33, 14, 25, 36

    tensor_sub(a3, b3);
    print_tensor("Test 4 Result", a3);  // Expected: 1, 2, 3, 4, 5, 6

    // Cleanup
    tensor_free(a1);
    tensor_free(b1);
    tensor_free(a2);
    tensor_free(b2);
    tensor_free(a3);
    tensor_free(b3);

    struct Tensor* x = tensor_from_data((float[]){1, 2, 3, 4, 5, 6}, (int[]){2, 3}, 2);

    print_tensor("Original X", x);
    tensor_transpose_2d(x);
    print_tensor("Transpose of X", x);
    tensor_reshape(x, (int[]){6}, 1);
    print_tensor("reshape of X", x);
    tensor_free(x);

    return 0;
}
