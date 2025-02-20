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
const int N = 32; // small size for rtl simulation

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h target_buffer = nullptr;
vx_buffer_h loss_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
        vx_mem_free(input_buffer);
        vx_mem_free(target_buffer);
        vx_mem_free(loss_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

void generate_random_data(float* data, int size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

float compute_mse_cpu(const float* inp, const float* y, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = inp[i] - y[i];
        sum += diff * diff;
    }
    return sum / N;
}

int main() {
    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_target(N);
    float h_loss = 0.0f;
    
    // Generate random data
    generate_random_data(h_input.data(), N);
    generate_random_data(h_target.data(), N);
    
    // Open device connection
    std::cout << "Opening device connection..." << std::endl;
    RT_CHECK(vx_dev_open(&device));
    
    // Get device capabilities (equivalent to CUDA device properties)
    uint64_t num_cores, num_warps, num_threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    
    std::cout << "Device info:" << std::endl;
    std::cout << "- Cores: " << num_cores << std::endl;
    std::cout << "- Warps: " << num_warps << std::endl;
    std::cout << "- Threads: " << num_threads << std::endl;
    
    // Allocate device memory (equivalent to cudaMalloc)
    std::cout << "Allocating device memory..." << std::endl;
    RT_CHECK(vx_mem_alloc(device, N * sizeof(float), VX_MEM_READ, &input_buffer));
    RT_CHECK(vx_mem_alloc(device, N * sizeof(float), VX_MEM_READ, &target_buffer));
    RT_CHECK(vx_mem_alloc(device, sizeof(float), VX_MEM_WRITE, &loss_buffer));
    
    // Get device addresses
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
    RT_CHECK(vx_mem_address(target_buffer, &kernel_arg.target_addr));
    RT_CHECK(vx_mem_address(loss_buffer, &kernel_arg.loss_addr));
    kernel_arg.N = N;
    
    // Copy data to device (equivalent to cudaMemcpy)
    std::cout << "Copying data to device..." << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, N * sizeof(float)));
    RT_CHECK(vx_copy_to_dev(target_buffer, h_target.data(), 0, N * sizeof(float)));
    
    // Upload kernel and arguments
    std::cout << "Uploading kernel..." << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
    
    // Launch kernel (equivalent to kernel<<<32,1>>>)
    std::cout << "Launching kernel..." << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    
    // Wait for completion (equivalent to cudaDeviceSynchronize)
    std::cout << "Waiting for completion..." << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    
    // Copy result back (equivalent to cudaMemcpy)
    RT_CHECK(vx_copy_from_dev(&h_loss, loss_buffer, 0, sizeof(float)));
    
    // Compute CPU reference result
    float cpu_loss = compute_mse_cpu(h_input.data(), h_target.data(), N);
    
    // Print results
    printf("GPU MSE Loss: %f\n", h_loss);
    printf("CPU MSE Loss: %f\n", cpu_loss);
    printf("Relative difference: %f%%\n", 100.0f * fabsf(h_loss - cpu_loss) / cpu_loss);
    
    // Verify results
    const float tolerance = 1e-3;
    if (fabsf(h_loss - cpu_loss) / cpu_loss < tolerance) {
        printf("Verification passed!\n");
    } else {
        printf("Verification failed!\n");
    }
    
    cleanup();
    return 0;
}