
#include <stdbool.h>

#include "tensor.h"

struct PicoLinear {
    int in_features;
    int out_features;
    struct PicoTensor* weights;  // Shape: [in_features, out_features]
    struct PicoTensor* bias;     // Shape: [out_features, 1]
};

struct PicoLinear* pico_nn_linear_init(int in_features, int out_features, bool bias);
struct PicoTensor* pico_nn_linear_forward(struct PicoLinear* layer, struct PicoTensor* input);
void pico_nn_linear_free(struct PicoLinear* linear);
