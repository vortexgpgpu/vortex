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
vx_buffer_h output_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
        vx_mem_free(input_buffer);
        vx_mem_free(output_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

// CPU reference reduction
float cpuReduce(const std::vector<float>& data) {
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    // Generate input data
    std::vector<float> h_input(N_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (unsigned int i = 0; i < N_SIZE; i++) {
        h_input[i] = dis(gen);
    }
    
    // Calculate CPU reference result
    float cpu_sum = cpuReduce(h_input);
    
    // Open device connection
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));
    
    // Calculate grid dimensions for smaller block size
    unsigned int gridSize = (N_SIZE + HW_THREADS_PER_CORE * 2 - 1) / (HW_THREADS_PER_CORE * 2);
    std::vector<float> h_output(gridSize);
    
    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    RT_CHECK(vx_mem_alloc(device, N_SIZE * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_alloc(device, gridSize * sizeof(float), VX_MEM_WRITE, &output_buffer));
    
    // Setup kernel arguments
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
    RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));
    kernel_arg.N = N_SIZE;
    
    // Copy input data to device
    std::cout << "Copying data to device..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, N_SIZE * sizeof(float)));
    
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
    
    // Copy result back
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, gridSize * sizeof(float)));
    
    // Sum up the partial results
    float gpu_sum = 0.0f;
    for (unsigned int i = 0; i < gridSize; i++) {
        gpu_sum += h_output[i];
    }

    std::cout<<"h_output:\n";
    for (int i = 0; i < gridSize; i++) {
        std::cout<<h_output[i]<<" ";
    }
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "Input size: " << h_input.size() << std::endl;
    std::cout << "Block size: " << HW_THREADS_PER_CORE << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "CPU sum = " << cpu_sum << std::endl;
    std::cout << "GPU sum = " << gpu_sum << std::endl;
    
    // Verify results
    float relative_error = std::abs(cpu_sum - gpu_sum) / std::abs(cpu_sum);
    std::cout << "Relative error = " << relative_error * 100.0f << "%" << std::endl;
    
    const float tolerance = 1e-5;
    if (relative_error < tolerance) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }
    
    cleanup();
    return 0;
}