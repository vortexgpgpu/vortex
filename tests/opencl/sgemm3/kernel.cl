#include "common.h"

__kernel void sgemm3(__global TYPE *A,
                     __global TYPE *B,
                     __global TYPE *C,
                     const unsigned int N,
                     __local TYPE *localA,
                     __local TYPE *localB)
{
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow  = get_local_id(1);
    int localCol  = get_local_id(0);
    int localSize = get_local_size(0);

    TYPE sum = 0;

    // Iterate over blocks
    for (int k = 0; k < N; k += localSize) {
        // Load block of matrix A & B to local memory
        localA[localRow * localSize + localCol] = A[globalRow * N + (k + localCol)];
        localB[localRow * localSize + localCol] = B[(k + localRow) * N + globalCol];

        // Synchronize to make sure the tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrix blocks and accumulate result
        for (int j = 0; j < localSize; j++) {
            sum += localA[localRow * localSize + j] * localB[j * localSize + localCol];
        }

        // Ensure computation is done before loading next block
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalCol] = sum;
}