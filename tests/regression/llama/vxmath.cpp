#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include "common.h"
#ifdef LLAMA_LEGACY_CYCLES
#include <VX_types.h>
static uint64_t legacy_launches = 0;
static uint64_t legacy_cycles = 0;
#endif

#ifdef DEBUG
    #define D(x) x
#else
    #define D(x)
#endif

#define RT_CHECK(_expr)                                         \
    do {                                                         \
        int _ret = _expr;                                          \
        if (0 == _ret)                                             \
            break;                                                   \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
        exit(-1);                                                  \
    } while (false)


////////////////////////////////////////////////////////////////////////////////
static uint64_t num_cores, num_warps, num_threads, num_total_threads;
vx_device_h device = nullptr;

void vx_init() {
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    num_total_threads = num_cores * num_warps * num_threads;
}

////////////////////////////////////////////////////////////////////////////////
// Optimized Matmul with buffer reuse, batching, and persistent kernel
static vx_buffer_h matmul_A_buffer = nullptr;
static vx_buffer_h matmul_B_buffer = nullptr;
static vx_buffer_h matmul_C_buffer = nullptr;
static vx_buffer_h matmul_krnl_buffer = nullptr;
static vx_buffer_h matmul_args_buffer = nullptr;
static matmul_kernel_args_t matmul_kernel_arg = {};
static char *kernel = "kernel.vxbin";

// Buffer size cache to avoid reallocation
static size_t cached_A_sz = 0;
static size_t cached_B_sz = 0;
static size_t cached_C_sz = 0;
static bool kernel_loaded = false;

// Batch operation support
struct MatmulBatch {
    float* A;
    float* B; 
    float* C;
    int M, N, K;
};

static std::vector<MatmulBatch> pending_ops;

void matmul_cleanup() {
    if (device) {
        if (matmul_A_buffer) vx_mem_free(matmul_A_buffer);
        if (matmul_B_buffer) vx_mem_free(matmul_B_buffer);
        if (matmul_C_buffer) vx_mem_free(matmul_C_buffer);
        if (matmul_krnl_buffer) vx_mem_free(matmul_krnl_buffer);
        if (matmul_args_buffer) vx_mem_free(matmul_args_buffer);
        
        matmul_A_buffer = matmul_B_buffer = matmul_C_buffer = nullptr;
        matmul_krnl_buffer = matmul_args_buffer = nullptr;
        cached_A_sz = cached_B_sz = cached_C_sz = 0;
        kernel_loaded = false;
        
        vx_dev_close(device);
        device = nullptr;
    }
}

// Optimized single matmul with buffer reuse
void vx_matmul_optimized(float* C, float* A, float* B, int M, int N, int K) {
    size_t A_sz = M * K * sizeof(float);
    size_t B_sz = K * N * sizeof(float);
    size_t C_sz = M * N * sizeof(float);

    // printf("vx_matmul_optimized: M=%d N=%d K=%d\n", M, N, K);

    // prepare kernel argument
    matmul_kernel_arg.M = M;
    matmul_kernel_arg.N = N;
    matmul_kernel_arg.K = K;

    const uint32_t BLOCK_SIZE = 4;
    matmul_kernel_arg.block_dim[0] = BLOCK_SIZE;
    matmul_kernel_arg.block_dim[1] = (N == 1) ? 1 : BLOCK_SIZE;
    matmul_kernel_arg.grid_dim[0] = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matmul_kernel_arg.grid_dim[1] = (N + matmul_kernel_arg.block_dim[1] - 1) / matmul_kernel_arg.block_dim[1];

    // Allocate or reuse buffer A (only if size changed)
    if (!matmul_A_buffer || A_sz > cached_A_sz) {
        if (matmul_A_buffer) vx_mem_free(matmul_A_buffer);
        RT_CHECK(vx_mem_alloc(device, A_sz, VX_MEM_READ, &matmul_A_buffer));
        cached_A_sz = A_sz;
    }
    RT_CHECK(vx_mem_address(matmul_A_buffer, &matmul_kernel_arg.A_addr));

    // Allocate or reuse buffer B
    if (!matmul_B_buffer || B_sz > cached_B_sz) {
        if (matmul_B_buffer) vx_mem_free(matmul_B_buffer);
        RT_CHECK(vx_mem_alloc(device, B_sz, VX_MEM_READ, &matmul_B_buffer));
        cached_B_sz = B_sz;
    }
    RT_CHECK(vx_mem_address(matmul_B_buffer, &matmul_kernel_arg.B_addr));

    // Allocate or reuse buffer C
    if (!matmul_C_buffer || C_sz > cached_C_sz) {
        if (matmul_C_buffer) vx_mem_free(matmul_C_buffer);
        RT_CHECK(vx_mem_alloc(device, C_sz, VX_MEM_WRITE, &matmul_C_buffer));
        cached_C_sz = C_sz;
    }
    RT_CHECK(vx_mem_address(matmul_C_buffer, &matmul_kernel_arg.C_addr));

    // Load kernel only once (persistent kernel)
    if (!kernel_loaded) {
        D(std::cout << "upload program (one-time)" << std::endl;)
        RT_CHECK(vx_upload_kernel_file(device, kernel, &matmul_krnl_buffer));
        kernel_loaded = true;
    }

    // Args buffer (small, always reallocate for simplicity)
    if (!matmul_args_buffer) {
        RT_CHECK(vx_mem_alloc(device, sizeof(matmul_kernel_args_t), VX_MEM_READ, &matmul_args_buffer));
    }

    // Upload data
    RT_CHECK(vx_copy_to_dev(matmul_A_buffer, A, 0, A_sz));
    RT_CHECK(vx_copy_to_dev(matmul_B_buffer, B, 0, B_sz));
    RT_CHECK(vx_copy_to_dev(matmul_args_buffer, &matmul_kernel_arg, 0, sizeof(matmul_kernel_args_t)));

    // Execute kernel
    RT_CHECK(vx_start(device, matmul_krnl_buffer, matmul_args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
#ifdef LLAMA_LEGACY_CYCLES
    {
        uint64_t cycles = 0;
        RT_CHECK(vx_mpm_query(device, VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE, 0, &cycles));
        legacy_cycles += cycles;
        ++legacy_launches;
    }
#endif

    // Download result
    RT_CHECK(vx_copy_from_dev(C, matmul_C_buffer, 0, C_sz));
}

void vx_matmul(float* C, float* A, float* B, int M, int N, int K) {
    vx_matmul_optimized(C, A, B, M, N, K);
}

// True batched operation for QKV computation using concatenated matrices
void vx_matmul_qkv_batch(float* Q, float* K, float* V, 
                         float* input, 
                         float* Wq, float* Wk, float* Wv,
                         int dim, int kv_dim) {
    // True batching: concatenate weight matrices vertically and do one large matmul
    // Wq is dim x dim, Wk is kv_dim x dim, Wv is kv_dim x dim
    // Concatenated: [(dim + kv_dim + kv_dim) x dim] @ [dim x 1] = [(dim + 2*kv_dim) x 1]
    
    int total_output_dim = dim + 2 * kv_dim;
    int input_dim = dim;
    
    // Allocate temporary concatenated weight matrix
    float* W_concat = (float*)malloc(total_output_dim * input_dim * sizeof(float));
    float* output_concat = (float*)malloc(total_output_dim * sizeof(float));
    
    // Copy Wq (dim x dim) to top of concatenated matrix
    memcpy(W_concat, Wq, dim * input_dim * sizeof(float));
    
    // Copy Wk (kv_dim x dim) to middle of concatenated matrix  
    memcpy(W_concat + dim * input_dim, Wk, kv_dim * input_dim * sizeof(float));
    
    // Copy Wv (kv_dim x dim) to bottom of concatenated matrix
    memcpy(W_concat + (dim + kv_dim) * input_dim, Wv, kv_dim * input_dim * sizeof(float));
    
    // Single batched matmul: W_concat @ input = output_concat
    vx_matmul_optimized(output_concat, W_concat, input, total_output_dim, 1, input_dim);
    
    // Split the concatenated output back into Q, K, V
    memcpy(Q, output_concat, dim * sizeof(float));                           // Q = first dim elements
    memcpy(K, output_concat + dim, kv_dim * sizeof(float));                  // K = next kv_dim elements  
    memcpy(V, output_concat + dim + kv_dim, kv_dim * sizeof(float));         // V = last kv_dim elements
    
    // Cleanup temporary buffers
    free(W_concat);
    free(output_concat);
}

// True batched operation for FFN w1/w3 computation using concatenated matrices
void vx_matmul_ffn_batch(float* out1, float* out2, float* input,
                         float* W1, float* W3, 
                         int input_dim, int hidden_dim) {
    // True batching: concatenate W1 and W3 vertically and do one large matmul
    // W1 is hidden_dim x input_dim, W3 is hidden_dim x input_dim
    // Concatenated: [(2*hidden_dim) x input_dim] @ [input_dim x 1] = [(2*hidden_dim) x 1]
    
    int total_output_dim = 2 * hidden_dim;
    
    // Allocate temporary concatenated weight matrix
    float* W_concat = (float*)malloc(total_output_dim * input_dim * sizeof(float));
    float* output_concat = (float*)malloc(total_output_dim * sizeof(float));
    
    // Copy W1 (hidden_dim x input_dim) to top half
    memcpy(W_concat, W1, hidden_dim * input_dim * sizeof(float));
    
    // Copy W3 (hidden_dim x input_dim) to bottom half
    memcpy(W_concat + hidden_dim * input_dim, W3, hidden_dim * input_dim * sizeof(float));
    
    // Single batched matmul: W_concat @ input = output_concat
    vx_matmul_optimized(output_concat, W_concat, input, total_output_dim, 1, input_dim);
    
    // Split the concatenated output back into out1 and out2
    memcpy(out1, output_concat, hidden_dim * sizeof(float));                 // out1 = first half
    memcpy(out2, output_concat + hidden_dim, hidden_dim * sizeof(float));    // out2 = second half
    
    // Cleanup temporary buffers
    free(W_concat);
    free(output_concat);
}

#ifdef LLAMA_LEGACY_CYCLES
void vx_legacy_report(int tokens) {
    double cpt = tokens > 0 ? (double)legacy_cycles / tokens : 0.0;
    double tps = cpt > 0 ? 400e6 / cpt : 0.0;
    printf("[LLAMA-LEGACY] tokens=%d launches=%lu device_cycles=%lu cycles_per_token=%.0f tok_per_s_400MHz=%.3f\n",
           tokens, (unsigned long)legacy_launches, (unsigned long)legacy_cycles, cpt, tps);
}
#endif
