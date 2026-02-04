#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// This kernel is designed for single-core parallel execution (scales with warps/threads). Multi-core would require inter-core synchronization between layers, which Vortex's vx_spawn_threads doesn't provide yet (to the best of my knowledge)...

// ReLU activation function
inline TYPE relu(TYPE x) {
    return (x > 0) ? x : 0;
}

// Layer execution arguments passed to spawned threads
typedef struct {
    TYPE* input;
    TYPE* weights;
    TYPE* bias;
    TYPE* output;
    uint32_t input_dim;
    uint32_t output_dim;
    bool apply_relu;
} layer_args_t;


// Softmax execution arguments
typedef struct {
    TYPE* data;
    uint32_t size;
    TYPE* max_val;
    TYPE* sum_exp;
} softmax_args_t;

// Global shared variables for softmax reductions
TYPE g_softmax_max_val;
TYPE g_softmax_sum_exp;

// Thread function: thread 0 finds max, others wait
void softmax_find_max_thread(softmax_args_t* __UNIFORM__ args) {
    if (blockIdx.x == 0) {
        // Thread 0 does the reduction serially
        TYPE max_val = args->data[0];
        for (uint32_t i = 1; i < args->size; ++i) {
            if (args->data[i] > max_val) {
                max_val = args->data[i];
            }
        }
        *(args->max_val) = max_val;
    }
}

// Thread function: compute exp in parallel, thread 0 sums
void softmax_exp_sum_thread(softmax_args_t* __UNIFORM__ args) {
    uint32_t i = blockIdx.x;
    
    // All threads compute their exp value
    if (i < args->size) {
        args->data[i] = expf(args->data[i] - *(args->max_val));
    }
    
    // Thread 0 does the sum reduction serially
    if (i == 0) {
        TYPE sum_exp = 0.0f;
        for (uint32_t j = 0; j < args->size; ++j) {
            sum_exp += args->data[j];
        }
        *(args->sum_exp) = sum_exp;
    }
}

// Thread function: normalize in parallel
void softmax_normalize_thread(softmax_args_t* __UNIFORM__ args) {
    uint32_t i = blockIdx.x;
    if (i < args->size) {
        args->data[i] /= *(args->sum_exp);
    }
}

// Softmax implementation
void apply_softmax(TYPE* data, uint32_t size) {
    softmax_args_t args;
    args.data = data;
    args.size = size;
    args.max_val = &g_softmax_max_val;
    args.sum_exp = &g_softmax_sum_exp;
    
    uint32_t single_thread = 1;
    vx_spawn_threads(1, &single_thread, nullptr, (vx_kernel_func_cb)softmax_find_max_thread, &args);

    vx_spawn_threads(1, &size, nullptr, (vx_kernel_func_cb)softmax_exp_sum_thread, &args);
    
    vx_spawn_threads(1, &size, nullptr, (vx_kernel_func_cb)softmax_normalize_thread, &args);
}

// Each thread computes one output neuron
// y[i] = sum_j(W[i][j] * x[j]) + b[i]
void fc_layer_thread(layer_args_t* __UNIFORM__ args) {
    uint32_t i = blockIdx.x;  // Output neuron index
    
    if (i >= args->output_dim) return;
    
    TYPE sum = 0;
    
    // Compute dot product: W[i] · input
    for (uint32_t j = 0; j < args->input_dim; ++j) {
        sum += args->weights[i * args->input_dim + j] * args->input[j];
    }
    
    // Add bias
    sum += args->bias[i];
    
    // Apply ReLU activation (except for output layer)
    if (args->apply_relu) {
        sum = relu(sum);
    }
    
    args->output[i] = sum;
}

// MLP inference kernel
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto layer_configs = reinterpret_cast<layer_config_t*>(arg->layer_configs_addr);
    
    // Layer buffers
    auto input = reinterpret_cast<TYPE*>(arg->input_addr);
    auto buffer1 = reinterpret_cast<TYPE*>(arg->buffer1_addr);
    auto buffer2 = reinterpret_cast<TYPE*>(arg->buffer2_addr);
    auto output = reinterpret_cast<TYPE*>(arg->output_addr);
    
    // Pointer arrays for ping-pong buffering
    TYPE* layer_inputs[4] = {input, buffer1, buffer2, buffer1};
    TYPE* layer_outputs[4] = {buffer1, buffer2, buffer1, output};
    
    layer_args_t layer_args;
    
    // Execute each layer with parallel threads
    for (uint32_t layer = 0; layer < arg->num_layers; ++layer) {
        auto cfg = &layer_configs[layer];
        
        layer_args.input = layer_inputs[layer];
        layer_args.weights = reinterpret_cast<TYPE*>(cfg->weights_addr);
        layer_args.bias = reinterpret_cast<TYPE*>(cfg->bias_addr);
        layer_args.output = layer_outputs[layer];
        layer_args.input_dim = cfg->input_dim;
        layer_args.output_dim = cfg->output_dim;
        layer_args.apply_relu = (layer < arg->num_layers - 1);
        
        // vx_spawn_threads provides barrier synchronization after all threads complete
        vx_spawn_threads(1, &layer_args.output_dim, nullptr, 
                        (vx_kernel_func_cb)fc_layer_thread, &layer_args);
    }
    
    // Apply softmax to final output
    uint32_t output_size = layer_configs[arg->num_layers - 1].output_dim;
    apply_softmax(output, output_size);
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    
    uint32_t grid_dim = 1;
    return vx_spawn_threads(1, &grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
