#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
int map_index(int global_i, int* t_shape, int* t_strides, int* res_strides, int ndim);
int* pad_shape(struct Tensor* smaller, int ndim);
int* pad_stride(struct Tensor* smaller, int ndim);
int check_broadcast_compatibility(struct Tensor* a, struct Tensor* b);

/* =========================================================================
   BACKEND DISPATCH (Internal)
   ========================================================================= */
struct Tensor* tensor_matmul_cpu_2d(struct Tensor* x, struct Tensor* y);
struct Tensor* tensor_matmul_cpu(struct Tensor* x, struct Tensor* y);
struct Tensor* tensor_dot_cpu(struct Tensor* x, struct Tensor* y);
struct Tensor* tensor_truediv_cpu(struct Tensor* x, struct Tensor* y);
struct Tensor* tensor_sub_cpu(struct Tensor* x, struct Tensor* y);
struct Tensor* tensor_add_cpu(struct Tensor* x, struct Tensor* y);

float tensor_max_cpu(struct Tensor* x);
float tensor_sum_cpu(struct Tensor* x);
float tensor_mean_cpu(struct Tensor* x);

// ops tables
static const struct TensorOps tensor_ops_cpu = {
    tensor_matmul_cpu, tensor_dot_cpu, tensor_truediv_cpu, tensor_sub_cpu,
    tensor_add_cpu,    tensor_max_cpu, tensor_sum_cpu,     tensor_mean_cpu};  // cpu functions

static const struct TensorOps tensor_ops_gpu = {
    tensor_matmul_cpu, tensor_dot_cpu, tensor_truediv_cpu, tensor_sub_cpu, tensor_add_cpu,
    tensor_max_cpu,    tensor_sum_cpu, tensor_mean_cpu};  // use cpu functions for now

/* =========================================================================
   Create tensors
   ========================================================================= */

struct Tensor* tensor_create(int* shape, int ndim) {
    struct Tensor* t;

    t = (struct Tensor*)malloc(sizeof(struct Tensor));
    if(t == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    t->ndim = ndim;

    t->shape = malloc(ndim * sizeof(int));
    t->strides = malloc(ndim * sizeof(int));

    for(int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }

    int total_elements = 1;
    for(int i = 0; i < ndim; i++) {
        total_elements *= t->shape[i];
    }
    t->data = malloc(total_elements * sizeof(float));

    for(int i = 0; i < total_elements; i++) {
        t->data[i] = 0.0f;
    }

    t->numel = total_elements;

    // AUTOGRAD INITIALIZATION
    t->requires_grad = 0;  // Default: no grad needed
    t->grad = NULL;        // Only allocate when requires_grad = 1
    t->grad_op = NONE;     // Leaf tensor by default
    t->num_parents = 0;
    t->parents = NULL;
    t->parents_values = NULL;

    tensor_update_strides(t);
    tensor_to_cpu(t);
    return t;
}
struct Tensor* tensor_from_data(float* existing_data, int* shape, int ndim) {
    struct Tensor* t = tensor_create(shape, ndim);
    if(t == NULL)
        return NULL;

    for(int i = 0; i < t->numel; i++) {
        t->data[i] = existing_data[i];
    }

    return t;
}
void tensor_free(struct Tensor* t) {
    free(t->data);
    free(t->shape);
    free(t->strides);

    // Free gradient if allocated
    if(t->grad != NULL) {
        free(t->grad);
    }

    // Free parents array (don't free the parent tensors themselves)
    if(t->parents != NULL) {
        free(t->parents);
    }

    // Free cached parent values
    if(t->parents_values != NULL) {
        for(int i = 0; i < t->num_parents; i++) {
            if(t->parents_values[i] != NULL) {
                free(t->parents_values[i]);
            }
        }
        free(t->parents_values);
    }
    free(t);
}

/* =========================================================================
   PUBLIC API (The Wrappers)
   ========================================================================= */
struct Tensor* tensor_add(struct Tensor* a, struct Tensor* b) {
    if(!check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "Error: Shapes are not broadcastable!\n");
        return NULL;
    }
    return a->ops->add(a, b);
}

struct Tensor* tensor_sub(struct Tensor* a, struct Tensor* b) {
    if(!check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "Error: Shapes are not broadcastable!\n");
        return NULL;
    }
    return a->ops->sub(a, b);
}

struct Tensor* tensor_truediv(struct Tensor* a, struct Tensor* b) {
    if(!check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "Error: Shapes are not broadcastable!\n");
        return NULL;
    }
    return a->ops->truediv(a, b);
}

struct Tensor* tensor_dot(struct Tensor* a, struct Tensor* b) {
    if(!check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "Error: Shapes are not broadcastable!\n");
        return NULL;
    }
    return a->ops->dot(a, b);
}

// NOTE: till i fix it use 2d for now
struct Tensor* tensor_matmul(struct Tensor* a, struct Tensor* b) {
    return tensor_matmul_cpu_2d(a, b);
}

struct Tensor* tensor_matmul_cpu(struct Tensor* x, struct Tensor* y) {
    return tensor_matmul_cpu_2d(x, y);  // for now
}

float tensor_mean(struct Tensor* a) {
    return a->ops->mean(a);
}
float tensor_sum(struct Tensor* a) {
    return a->ops->sum(a);
}
float tensor_max(struct Tensor* a) {
    return a->ops->max(a);
}

void tensor_reshape(struct Tensor* a, int* shape, int ndim) {
    int num_el = 1;
    for(int x = 0; x < ndim; x++) {
        num_el *= shape[x];
    }
    if(num_el != a->numel) {
        fprintf(stderr, "Error: Invalid reshape!\n");
        return;
    }
    a->shape = realloc(a->shape, sizeof(int) * ndim);
    a->strides = realloc(a->strides, sizeof(int) * ndim);
    a->ndim = ndim;

    for(int i = 0; i < ndim; i++) {
        a->shape[i] = shape[i];
    }

    tensor_update_strides(a);
}

void tensor_transpose_2d(struct Tensor* a) {
    if(a->ndim != 2) {
        fprintf(stderr, "Error: This is not a rank 2 tensor!\n");
        return;
    }

    // swap the r and c
    int c = a->shape[1];
    a->shape[1] = a->shape[0];
    a->shape[0] = c;

    int sc = a->strides[1];
    a->strides[1] = a->strides[0];
    a->strides[0] = sc;
}

struct Tensor* tensor_clone(struct Tensor* a) {
    struct Tensor* t;

    t = (struct Tensor*)malloc(sizeof(struct Tensor));
    if(t == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    t->ndim = a->ndim;
    t->shape = malloc(a->ndim * sizeof(int));
    t->strides = malloc(a->ndim * sizeof(int));

    for(int i = 0; i < a->ndim; i++) {
        t->shape[i] = a->shape[i];
    }
    t->numel = a->numel;
    t->data = malloc(a->numel * sizeof(float));

    for(int i = 0; i < a->numel; i++) {
        t->data[i] = a->data[i];
    }

    // AUTOGRA SETTINGS
    t->requires_grad = a->requires_grad;
    t->grad = NULL;
    t->grad_op = NONE;
    t->num_parents = 0;
    t->parents = NULL;
    t->parents_values = NULL;

    tensor_update_strides(t);
    tensor_to_cpu(t);

    return t;
};

/* =========================================================================
   CPU BACKEND IMPLEMENTATION
   ========================================================================= */

struct Tensor* tensor_matmul_cpu_2d(struct Tensor* x, struct Tensor* y) {
    if(x->ndim != 2 || y->ndim != 2) {
        perror("2d matmul matrices must both be in the 2nd rank");
        return NULL;
    }

    if(x->shape[1] != y->shape[0]) {
        perror("2s matmul matrices must be compatible");
        return NULL;
    }

    int* res_shape = malloc(sizeof(int) * 2);

    /*
        x = [M , N]
        Y = [N , P]
        r = [M , P]
    */

    int rows = x->shape[0];     // M
    int columns = y->shape[1];  // P
    int k_dim = x->shape[1];    // N

    res_shape[0] = rows;
    res_shape[1] = columns;

    struct Tensor* res = tensor_create(res_shape, 2);
    tensor_update_strides(res);

    for(int i = 0; i < rows; i++) {
        for(int k = 0; k < k_dim; k++) {
            float m_cell = x->data[i * x->strides[0] + k * x->strides[1]];

            for(int j = 0; j < columns; j++) {
                res->data[i * res->strides[0] + j * res->strides[1]] +=
                    m_cell * y->data[k * y->strides[0] + j * y->strides[1]];
            }
        }
    }

    // AUTOGRAD: Store operation info
    res->grad_op = MATMUL;
    res->num_parents = 2;
    res->parents = malloc(sizeof(struct Tensor*) * 2);
    res->parents[0] = x;
    res->parents[1] = y;
    res->requires_grad = x->requires_grad || y->requires_grad;

    // Cache parent values for backward pass
    res->parents_values = malloc(sizeof(float*) * 2);
    res->parents_values[0] = malloc(sizeof(float) * x->numel);
    res->parents_values[1] = malloc(sizeof(float) * y->numel);

    for(int i = 0; i < x->numel; i++)
        res->parents_values[0][i] = x->data[i];
    for(int i = 0; i < y->numel; i++)
        res->parents_values[1][i] = y->data[i];

    return res;
};

struct Tensor* tensor_dot_cpu(struct Tensor* x, struct Tensor* y) {
    int ndim = MAX(x->ndim, y->ndim);
    int require_grad = MAX(x->ndim, y->ndim);

    int* x_padded_shape = pad_shape(x, ndim);
    int* x_padded_strides = pad_stride(x, ndim);
    int* y_padded_shape = pad_shape(y, ndim);
    int* y_padded_strides = pad_stride(y, ndim);

    int* res_shape = malloc(sizeof(int) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(x_padded_shape[i], y_padded_shape[i]);

    struct Tensor* res = tensor_create(res_shape, ndim);
    tensor_update_strides(res);

    for(int i = 0; i < res->numel; i++) {
        int ix = map_index(i, x_padded_shape, x_padded_strides, res->strides, ndim);
        int iy = map_index(i, y_padded_shape, y_padded_strides, res->strides, ndim);
        res->data[i] = x->data[ix] * y->data[iy];
    }

    // AUTOGRAD: Store operation info
    res->grad_op = MUL;
    res->num_parents = 2;
    res->parents = malloc(sizeof(struct Tensor*) * 2);
    res->parents[0] = x;
    res->parents[1] = y;
    res->requires_grad = require_grad;

    // Cache parent values for backward pass
    res->parents_values = malloc(sizeof(float*) * 2);
    res->parents_values[0] = malloc(sizeof(float) * x->numel);
    res->parents_values[1] = malloc(sizeof(float) * y->numel);
    for(int i = 0; i < x->numel; i++)
        res->parents_values[0][i] = x->data[i];
    for(int i = 0; i < y->numel; i++)
        res->parents_values[1][i] = y->data[i];

    free(x_padded_shape);
    free(x_padded_strides);
    free(y_padded_shape);
    free(y_padded_strides);
    free(res_shape);

    return res;
}

struct Tensor* tensor_truediv_cpu(struct Tensor* x, struct Tensor* y) {
    int ndim = MAX(x->ndim, y->ndim);
    int require_grad = MAX(x->ndim, y->ndim);

    int* x_padded_shape = pad_shape(x, ndim);
    int* x_padded_strides = pad_stride(x, ndim);
    int* y_padded_shape = pad_shape(y, ndim);
    int* y_padded_strides = pad_stride(y, ndim);

    int* res_shape = malloc(sizeof(int) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(x_padded_shape[i], y_padded_shape[i]);

    struct Tensor* res = tensor_create(res_shape, ndim);
    tensor_update_strides(res);

    for(int i = 0; i < res->numel; i++) {
        int ix = map_index(i, x_padded_shape, x_padded_strides, res->strides, ndim);
        int iy = map_index(i, y_padded_shape, y_padded_strides, res->strides, ndim);
        res->data[i] = x->data[ix] / y->data[iy];
    }

    // AUTOGRAD: Store operation info (add DIV to your enum!)
    res->grad_op = DIV;
    res->num_parents = 2;
    res->parents = malloc(sizeof(struct Tensor*) * 2);
    res->parents[0] = x;
    res->parents[1] = y;
    res->requires_grad = require_grad;

    // Cache parent values for backward pass
    res->parents_values = malloc(sizeof(float*) * 2);
    res->parents_values[0] = malloc(sizeof(float) * x->numel);
    res->parents_values[1] = malloc(sizeof(float) * y->numel);
    for(int i = 0; i < x->numel; i++)
        res->parents_values[0][i] = x->data[i];
    for(int i = 0; i < y->numel; i++)
        res->parents_values[1][i] = y->data[i];

    free(x_padded_shape);
    free(x_padded_strides);
    free(y_padded_shape);
    free(y_padded_strides);
    free(res_shape);

    return res;
}

struct Tensor* tensor_sub_cpu(struct Tensor* x, struct Tensor* y) {
    int ndim = MAX(x->ndim, y->ndim);
    int require_grad = MAX(x->ndim, y->ndim);

    int* x_padded_shape = pad_shape(x, ndim);
    int* x_padded_strides = pad_stride(x, ndim);
    int* y_padded_shape = pad_shape(y, ndim);
    int* y_padded_strides = pad_stride(y, ndim);

    int* res_shape = malloc(sizeof(int) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(x_padded_shape[i], y_padded_shape[i]);

    struct Tensor* res = tensor_create(res_shape, ndim);
    tensor_update_strides(res);

    for(int i = 0; i < res->numel; i++) {
        int ix = map_index(i, x_padded_shape, x_padded_strides, res->strides, ndim);
        int iy = map_index(i, y_padded_shape, y_padded_strides, res->strides, ndim);
        res->data[i] = x->data[ix] - y->data[iy];
    }

    // AUTOGRAD: Store operation info
    res->grad_op = SUB;
    res->num_parents = 2;
    res->parents = malloc(sizeof(struct Tensor*) * 2);
    res->parents[0] = x;
    res->parents[1] = y;
    res->requires_grad = require_grad;

    // Cache parent values for backward pass
    res->parents_values = malloc(sizeof(float*) * 2);
    res->parents_values[0] = malloc(sizeof(float) * x->numel);
    res->parents_values[1] = malloc(sizeof(float) * y->numel);
    for(int i = 0; i < x->numel; i++)
        res->parents_values[0][i] = x->data[i];
    for(int i = 0; i < y->numel; i++)
        res->parents_values[1][i] = y->data[i];

    free(x_padded_shape);
    free(x_padded_strides);
    free(y_padded_shape);
    free(y_padded_strides);
    free(res_shape);

    return res;
}

struct Tensor* tensor_add_cpu(struct Tensor* x, struct Tensor* y) {
    int ndim = MAX(x->ndim, y->ndim);
    int require_grad = MAX(x->ndim, y->ndim);

    int* x_padded_shape = pad_shape(x, ndim);
    int* x_padded_strides = pad_stride(x, ndim);
    int* y_padded_shape = pad_shape(y, ndim);
    int* y_padded_strides = pad_stride(y, ndim);

    int* res_shape = malloc(sizeof(int) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(x_padded_shape[i], y_padded_shape[i]);

    struct Tensor* res = tensor_create(res_shape, ndim);
    tensor_update_strides(res);

    for(int i = 0; i < res->numel; i++) {
        int ix = map_index(i, x_padded_shape, x_padded_strides, res->strides, ndim);
        int iy = map_index(i, y_padded_shape, y_padded_strides, res->strides, ndim);
        res->data[i] = x->data[ix] + y->data[iy];
    }

    // AUTOGRAD: Store operation info
    res->grad_op = ADD;
    res->num_parents = 2;
    res->parents = malloc(sizeof(struct Tensor*) * 2);
    res->parents[0] = x;
    res->parents[1] = y;
    res->requires_grad = require_grad;

    // AUTOGRAD: Cache parent values for backward pass
    res->parents_values = malloc(sizeof(float*) * 2);
    res->parents_values[0] = malloc(sizeof(float) * x->numel);
    res->parents_values[1] = malloc(sizeof(float) * y->numel);
    for(int i = 0; i < x->numel; i++)
        res->parents_values[0][i] = x->data[i];
    for(int i = 0; i < y->numel; i++)
        res->parents_values[1][i] = y->data[i];

    free(x_padded_shape);
    free(x_padded_strides);
    free(y_padded_shape);
    free(y_padded_strides);
    free(res_shape);

    return res;  // Return new tensor instead of mutating
}

// INFO: i don't think max is something that's called a lot
//       or else binary search would have been beatiful for this, but yh

float tensor_max_cpu(struct Tensor* t) {
    float max = t->data[0];
    for(int i = 1; i < t->numel; i++) {
        if(t->data[i] > max) {
            max = t->data[i];
        }
    }
    return max;
};

float tensor_sum_cpu(struct Tensor* t) {
    float sum = 0;
    for(int i = 0; i < t->numel; i++) {
        sum += t->data[i];
    }
    return sum;
};

float tensor_mean_cpu(struct Tensor* t) {
    return tensor_sum(t) / t->numel;
};

/* =========================================================================
   COMMONS
   ========================================================================= */

int* pad_shape(struct Tensor* smaller, int ndim) {
    int* padded = malloc(sizeof(int) * ndim);
    int diff = ndim - smaller->ndim;
    for(int i = 0; i < ndim; i++) {
        if(i < diff) {
            padded[i] = 1;
        } else {
            // subtract 'diff' to map the large index back to the small one
            padded[i] = smaller->shape[i - diff];
        }
    }
    return padded;
}

int* pad_stride(struct Tensor* smaller, int ndim) {
    int* padded = malloc(sizeof(int) * ndim);
    int diff = ndim - smaller->ndim;

    for(int i = 0; i < ndim; i++) {
        if(i < diff) {
            padded[i] = 0;
        } else {
            padded[i] = smaller->strides[i - diff];
        }
    }

    return padded;
}

int map_index(int global_i, int* t_shape, int* t_strides, int* res_strides, int ndim) {
    int mapped_idx = 0;
    int rem = global_i;
    for(int d = 0; d < ndim; d++) {
        int coord = rem / res_strides[d];
        rem %= res_strides[d];

        // If shape is 1, coord becomes 0. If shape > 1, coord stays coord.
        // This is the "stretching" logic.
        if(t_shape[d] > 1) {
            mapped_idx += coord * t_strides[d];
        }
    }
    return mapped_idx;
}

int check_broadcast_compatibility(struct Tensor* a, struct Tensor* b) {
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;

    // We check from the end of the shape arrays (the "trailing" dimensions)
    int i = ndim_a - 1;
    int j = ndim_b - 1;

    while(i >= 0 && j >= 0) {
        int dim_a = a->shape[i];
        int dim_b = b->shape[j];

        // The Broadcasting Rule:
        // 1. Dimensions are equal, OR
        // 2. One of them is 1
        if(dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return 0;  // Not compatible!
        }
        i--;
        j--;
    }

    // If one tensor has more dimensions (e.g., [5, 4, 3] vs [4, 3]),
    // the extra leading dimensions [5] are always compatible with
    // the "implicit ones" of the smaller tensor.
    return 1;
}

/* =========================================================================
   CORE UTILITIES & HELPERS
   ========================================================================= */
int tensor_ndim(struct Tensor* t) {
    return t->ndim;
}
int* tensor_shape(struct Tensor* t) {
    return t->shape;
}
float tensor_get_nd(struct Tensor* t, int* coords) {
    int final_index = 0;
    for(int i = 0; i < t->ndim; i++) {
        final_index += coords[i] * t->strides[i];
    }
    return t->data[final_index];
}
void tensor_update_strides(struct Tensor* t) {
    int current_stride = 1;
    for(int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = current_stride;
        current_stride *= t->shape[i];
    }
}

void print_tensor(const char* label, struct Tensor* t) {
    printf("%s (shape ", label);
    for(int i = 0; i < t->ndim; i++)
        printf("%d ", t->shape[i]);
    printf("):\n");

    // We still loop through total number of elements
    for(int i = 0; i < t->numel; i++) {
        int physical_idx = 0;

        int* logical_strides = malloc(sizeof(int) * t->ndim);
        int stride = 1;
        for(int d = t->ndim - 1; d >= 0; d--) {
            logical_strides[d] = stride;
            stride *= t->shape[d];
        }

        int temp_i = i;
        for(int d = 0; d < t->ndim; d++) {
            int coord = temp_i / logical_strides[d];
            temp_i %= logical_strides[d];

            // Map logical coord to physical memory using ACTUAL strides
            physical_idx += coord * t->strides[d];
        }

        printf("%.1f ", t->data[physical_idx]);

        // Fancy formatting: add a newline at the end of every "row"
        if(t->ndim > 1 && (i + 1) % t->shape[t->ndim - 1] == 0) {
            printf("\n");
        }

        free(logical_strides);
    }
    printf("\n");
}

/* =========================================================================
   DEVICE MANAGEMENT
   ========================================================================= */
void tensor_to_cpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_cpu;
}
void tensor_to_gpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_gpu;
}

int max(int x, int y) {
    if(x > y) {
        return x;
    } else {
        return y;
    }
}
