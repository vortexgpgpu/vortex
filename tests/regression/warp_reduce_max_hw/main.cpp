#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <random>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                  \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_vals_buffer = nullptr;
vx_buffer_h output_indices_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
        vx_mem_free(input_buffer);
        vx_mem_free(output_vals_buffer);
        vx_mem_free(output_indices_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

void generate_random_data(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }
}

void verify_results(const std::vector<float>& input, const std::vector<float>& output_vals, 
                   const std::vector<int>& output_indices) {
    const int num_warps = (ARRAY_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    
    for (int i = 0; i < num_warps; i++) {
        float max_val = input[i * WARP_SIZE];
        int max_idx = i * WARP_SIZE;
        
        // Find max in this warp
        for (int j = 1; j < WARP_SIZE; j++) {
            int idx = i * WARP_SIZE + j;
            if (idx < ARRAY_SIZE && input[idx] > max_val) {
                max_val = input[idx];
                max_idx = idx;
            }
        }
        
        // Verify results
        if (fabs(output_vals[i] - max_val) > 1e-6) {
            printf("Error: Warp %d max value mismatch. Expected %f, got %f\n",
                   i, max_val, output_vals[i]);
            return;
        }
        if (output_indices[i] != max_idx) {
            printf("Error: Warp %d max index mismatch. Expected %d, got %d\n",
                   i, max_idx, output_indices[i]);
            return;
        }
    }
    printf("All tests passed successfully!\n");
}

int main() {
    // Allocate host memory
    std::vector<float> h_input(ARRAY_SIZE);
    const int total_warps = (ARRAY_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    std::vector<float> h_output_vals(total_warps);
    std::vector<int> h_output_indices(total_warps);
    
    // Generate random input data
    generate_random_data(h_input);
    
    // Open device connection
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));
    
    // Get device capabilities
    uint64_t num_cores, num_warps, num_threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    
    std::cout << "Device info:" << std::endl;
    std::cout << "- Cores: " << num_cores << std::endl;
    std::cout << "- Warps: " << num_warps << std::endl;
    std::cout << "- Threads: " << num_threads << std::endl;
    
    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    RT_CHECK(vx_mem_alloc(device, ARRAY_SIZE * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_alloc(device, total_warps * sizeof(float), VX_MEM_WRITE, &output_vals_buffer));
    RT_CHECK(vx_mem_alloc(device, total_warps * sizeof(int), VX_MEM_WRITE, &output_indices_buffer));
    
    // Setup kernel arguments
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
    RT_CHECK(vx_mem_address(output_vals_buffer, &kernel_arg.output_vals_addr));
    RT_CHECK(vx_mem_address(output_indices_buffer, &kernel_arg.output_indices_addr));
    kernel_arg.N = ARRAY_SIZE;
    
    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, ARRAY_SIZE * sizeof(float)));
    
    // Upload kernel and arguments
    std::cout << "Uploading kernel..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
    
    // Launch kernel
    std::cout << "Launching kernel..." << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    
    // Wait for completion
    std::cout << "Waiting for completion..." << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    
    // Copy results back
    RT_CHECK(vx_copy_from_dev(h_output_vals.data(), output_vals_buffer, 0, total_warps * sizeof(float)));
    RT_CHECK(vx_copy_from_dev(h_output_indices.data(), output_indices_buffer, 0, total_warps * sizeof(int)));
    
    // Verify results
    verify_results(h_input, h_output_vals, h_output_indices);
    
    cleanup();
    return 0;
}