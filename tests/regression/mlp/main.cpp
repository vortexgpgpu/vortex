#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 150  // Higher tolerance for deep network (4 layers, accumulated FP errors)

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

// Device handles
vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h buffer1 = nullptr;
vx_buffer_h buffer2 = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h layer_configs_buffer = nullptr;

// Weight and bias buffers for each layer
vx_buffer_h weights_buffers[NUM_LAYERS];
vx_buffer_h bias_buffers[NUM_LAYERS];

kernel_arg_t kernel_arg = {};

// Layer dimensions
const uint32_t layer_input_dims[NUM_LAYERS] = {INPUT_DIM, HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM};
const uint32_t layer_output_dims[NUM_LAYERS] = {HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, OUTPUT_DIM};

static void show_usage() {
   std::cout << "Vortex MLP Test." << std::endl;
//    std::cout << "Usage: [-k: kernel] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(input_buffer);
    vx_mem_free(output_buffer);
    vx_mem_free(buffer1);
    vx_mem_free(buffer2);
    vx_mem_free(layer_configs_buffer);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      vx_mem_free(weights_buffers[i]);
      vx_mem_free(bias_buffers[i]);
    }
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// Initialize weights with Xavier initialization
void init_weights(std::vector<TYPE>& weights, uint32_t fan_in, uint32_t fan_out) {
    TYPE scale = std::sqrt(2.0f / (fan_in + fan_out));
    for (size_t i = 0; i < weights.size(); ++i) {
        // Generate random value between -scale and scale
        weights[i] = (static_cast<TYPE>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// Initialize biases to zero
void init_bias(std::vector<TYPE>& bias) {
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = 0.0f;
    }
}

// CPU reference implementation of ReLU
TYPE relu_cpu(TYPE x) {
    return (x > 0) ? x : 0;
}

// CPU reference implementation of fully connected layer
void fc_layer_cpu(const TYPE* input, const TYPE* weights, const TYPE* bias,
                  TYPE* output, uint32_t input_dim, uint32_t output_dim, bool apply_relu) {
    for (uint32_t i = 0; i < output_dim; ++i) {
        TYPE sum = 0;
        for (uint32_t j = 0; j < input_dim; ++j) {
            sum += weights[i * input_dim + j] * input[j];
        }
        sum += bias[i];
        output[i] = apply_relu ? relu_cpu(sum) : sum;
    }
}

// CPU reference implementation of softmax
void softmax_cpu(TYPE* data, uint32_t size) {
    // Find max for numerical stability
    TYPE max_val = data[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    // Compute exp(x - max) and sum
    TYPE sum_exp = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum_exp += data[i];
    }
    
    // Normalize
    for (uint32_t i = 0; i < size; ++i) {
        data[i] /= sum_exp;
    }
}

// Run MLP inference on CPU for reference
void mlp_cpu(const TYPE* input, TYPE* output,
             const std::vector<std::vector<TYPE>>& weights,
             const std::vector<std::vector<TYPE>>& biases) {
    
    std::vector<TYPE> layer_output[NUM_LAYERS];
    
    for (int l = 0; l < NUM_LAYERS; ++l) {
        layer_output[l].resize(layer_output_dims[l]);
        
        const TYPE* layer_input = (l == 0) ? input : layer_output[l-1].data();
        bool apply_relu = (l < NUM_LAYERS - 1);
        
        fc_layer_cpu(layer_input, weights[l].data(), biases[l].data(),
                    layer_output[l].data(), layer_input_dims[l], layer_output_dims[l], apply_relu);
    }
    
    // Copy final output
    memcpy(output, layer_output[NUM_LAYERS-1].data(), OUTPUT_DIM * sizeof(TYPE));
    
    // Apply softmax to final output
    softmax_cpu(output, OUTPUT_DIM);
}

// Compare floating point values with ULP tolerance
bool compare_float(TYPE a, TYPE b, int index, int& errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
        if (errors < 100) {
            printf("*** error: [%d] expected=%f, actual=%f (diff=%d ULP)\n", index, b, a, d);
        }
        return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);

    std::srand(42);  // Fixed seed for reproducible results across config changes

    std::cout << "=== Vortex MLP Neural Network Test ===" << std::endl;
    std::cout << "Network: " << INPUT_DIM << " -> " << HIDDEN1_DIM 
              << " -> " << HIDDEN2_DIM << " -> " << HIDDEN3_DIM 
              << " -> " << OUTPUT_DIM << std::endl;

    // Open device connection
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));

    // Allocate host buffers for weights and biases
    std::vector<std::vector<TYPE>> h_weights(NUM_LAYERS);
    std::vector<std::vector<TYPE>> h_biases(NUM_LAYERS);
    
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim = layer_input_dims[l];
        uint32_t out_dim = layer_output_dims[l];
        
        h_weights[l].resize(out_dim * in_dim);
        h_biases[l].resize(out_dim);
        
        init_weights(h_weights[l], in_dim, out_dim);
        init_bias(h_biases[l]);
    }

    // Allocate host input/output buffers
    std::vector<TYPE> h_input(INPUT_DIM);
    std::vector<TYPE> h_output(OUTPUT_DIM);
    std::vector<TYPE> h_ref_output(OUTPUT_DIM);

    // Initialize input with random values
    for (uint32_t i = 0; i < INPUT_DIM; ++i) {
        h_input[i] = static_cast<TYPE>(rand()) / RAND_MAX;
    }

    // Print input summary
    // std::cout << "Input (first 8 values): ";
    // for (int i = 0; i < 8; ++i) {
    //     std::cout << h_input[i] << " ";
    // }
    // std::cout << "..." << std::endl;

    // Allocate device memory for input/output
    std::cout << "Allocating device memory..." << std::endl;
    
    RT_CHECK(vx_mem_alloc(device, INPUT_DIM * sizeof(TYPE), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
    
    RT_CHECK(vx_mem_alloc(device, OUTPUT_DIM * sizeof(TYPE), VX_MEM_READ_WRITE, &output_buffer));
    RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));
    
    // Intermediate buffers (sized for largest hidden layer)
    uint32_t max_hidden = std::max(HIDDEN1_DIM, std::max(HIDDEN2_DIM, HIDDEN3_DIM));
    RT_CHECK(vx_mem_alloc(device, max_hidden * sizeof(TYPE), VX_MEM_READ_WRITE, &buffer1));
    RT_CHECK(vx_mem_address(buffer1, &kernel_arg.buffer1_addr));
    
    RT_CHECK(vx_mem_alloc(device, max_hidden * sizeof(TYPE), VX_MEM_READ_WRITE, &buffer2));
    RT_CHECK(vx_mem_address(buffer2, &kernel_arg.buffer2_addr));

    // Allocate and upload weights and biases
    std::vector<layer_config_t> layer_configs(NUM_LAYERS);
    
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t in_dim = layer_input_dims[l];
        uint32_t out_dim = layer_output_dims[l];
        
        // Allocate weights
        RT_CHECK(vx_mem_alloc(device, out_dim * in_dim * sizeof(TYPE), VX_MEM_READ, &weights_buffers[l]));
        RT_CHECK(vx_mem_address(weights_buffers[l], &layer_configs[l].weights_addr));
        RT_CHECK(vx_copy_to_dev(weights_buffers[l], h_weights[l].data(), 0, out_dim * in_dim * sizeof(TYPE)));
        
        // Allocate biases
        RT_CHECK(vx_mem_alloc(device, out_dim * sizeof(TYPE), VX_MEM_READ, &bias_buffers[l]));
        RT_CHECK(vx_mem_address(bias_buffers[l], &layer_configs[l].bias_addr));
        RT_CHECK(vx_copy_to_dev(bias_buffers[l], h_biases[l].data(), 0, out_dim * sizeof(TYPE)));
        
        layer_configs[l].input_dim = in_dim;
        layer_configs[l].output_dim = out_dim;
        
        std::cout << "  Layer " << l << ": " << in_dim << " x " << out_dim 
                  << " (weights: " << (out_dim * in_dim * sizeof(TYPE)) << " bytes)" << std::endl;
    }

    // Upload layer configs
    RT_CHECK(vx_mem_alloc(device, NUM_LAYERS * sizeof(layer_config_t), VX_MEM_READ, &layer_configs_buffer));
    RT_CHECK(vx_mem_address(layer_configs_buffer, &kernel_arg.layer_configs_addr));
    RT_CHECK(vx_copy_to_dev(layer_configs_buffer, layer_configs.data(), 0, NUM_LAYERS * sizeof(layer_config_t)));

    // Upload input
    std::cout << "Uploading input data..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, INPUT_DIM * sizeof(TYPE)));

    // Set kernel arguments
    kernel_arg.num_layers = NUM_LAYERS;
    kernel_arg.batch_size = 1;

    // Upload kernel binary
    std::cout << "Uploading kernel binary..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // Upload kernel arguments
    std::cout << "Uploading kernel arguments..." << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    // Start device
    std::cout << "Starting MLP inference on device..." << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();
    
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // Wait for completion
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    // Download output
    std::cout << "Downloading output..." << std::endl;
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, OUTPUT_DIM * sizeof(TYPE)));

    // Compute CPU reference
    std::cout << "Computing CPU reference..." << std::endl;
    mlp_cpu(h_input.data(), h_ref_output.data(), h_weights, h_biases);

    // Print outputs
    // std::cout << "Device output: ";
    // for (int i = 0; i < OUTPUT_DIM; ++i) {
    //     std::cout << h_output[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "CPU reference: ";
    // for (int i = 0; i < OUTPUT_DIM; ++i) {
    //     std::cout << h_ref_output[i] << " ";
    // }
    // std::cout << std::endl;

    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < OUTPUT_DIM; ++i) {
        if (!compare_float(h_output[i], h_ref_output[i], i, errors)) {
            ++errors;
        }
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    cleanup();

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
