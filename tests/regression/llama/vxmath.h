
// Initialize Vortex library
void vx_init();

// Perform matrix multiplication on Vortex
void vx_matmul(float* C, float* A, float* B, int M, int N, int K);

// Cleanup Vortex resources
void matmul_cleanup();

// Batched QKV computation for transformer layers
void vx_matmul_qkv_batch(float* Q, float* K, float* V, 
                         float* input, 
                         float* Wq, float* Wk, float* Wv,
                         int dim, int kv_dim);

// Batched FFN w1/w3 computation 
void vx_matmul_ffn_batch(float* out1, float* out2, float* input,
                         float* W1, float* W3, 
                         int input_dim, int hidden_dim);

#ifdef LLAMA_LEGACY_CYCLES
// Device-cycle accounting for the legacy port: MCYCLE is reset on every launch, so it is
// read after each vx_ready_wait and summed. tokens = number of forward() calls.
void vx_legacy_report(int tokens);
#endif
