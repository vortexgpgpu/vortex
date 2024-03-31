#include "common.h"

__kernel void sgemm2(__global float *A, 
                     __global float *B, 
                     __global float *C, 
                     const unsigned int N)
{
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow  = get_local_id(1);
    int localCol  = get_local_id(0);

    // Static local memory declaration
    __local float localA[LOCAL_SIZE][LOCAL_SIZE];
    __local float localB[LOCAL_SIZE][LOCAL_SIZE];

    float sum = 0.0f;

    // Iterate over blocks
    for (int k = 0; k < N; k += LOCAL_SIZE) {
        // Load a block of matrix A into local memory
        localA[localRow][localCol] = A[globalRow * N + k + localCol];

        // Load a block of matrix B into local memory
        localB[localRow][localCol] = B[(k + localRow) * N + globalCol];

        // Ensure the entire block is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute multiplication for this block
        for (int j = 0; j < LOCAL_SIZE; j++) {
            sum += localA[localRow][j] * localB[j][localCol];
        }

        // Ensure computation is done before loading next block
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalCol] = sum;
}