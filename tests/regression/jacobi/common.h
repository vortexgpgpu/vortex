#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
/*#define TYPE float*/
#define TYPE float
#endif

typedef struct {
  uint32_t size;

  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t src2_addr;
  uint64_t dst_addr;  
} kernel_arg_t;


void jacobi_cpu(float* A,       // flattened matrix A (size*size)
                float* x_old,   // previous iteration values
                float* x_new,   // output / current solution
                float* b,       // right-hand side
                int size,
                int max_iters) {

    for (int iter = 0; iter < max_iters; ++iter) {
        // Copy current solution into x_old
        for (int i = 0; i < size; ++i) x_old[i] = x_new[i];

        // Perform Jacobi update
        for (int i = 0; i < size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < size; ++j) {
                if (i != j) sum += A[i * size + j] * x_old[j];
            }
            x_new[i] = (b[i] - sum) / A[i * size + i];
        }


        // Print result after this iteration
        /*std::cout << "Iteration " << iter + 1 << ": ";*/
        /*for (int i = 0; i < size; ++i)*/
            /*std::cout << std::fixed << std::setprecision(6) << h_dst[i] << " ";*/
        /*std::cout << "\n";*/
    }
}



#endif
