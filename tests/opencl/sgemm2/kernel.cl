#include "common.h"

__kernel void sgemm2(__global TYPE *A,
                     __global TYPE *B,
                     __global TYPE *C,
                     const unsigned int N)
{
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow  = get_local_id(1);
    int localCol  = get_local_id(0);

    // Static local memory declaration
    __local TYPE localA[TILE_SIZE][TILE_SIZE];
    __local TYPE localB[TILE_SIZE][TILE_SIZE];

    TYPE sum = 0;

    // Iterate over blocks
    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load block of matrix A & B to local memory
        localA[localRow][localCol] = A[globalRow * N + (k + localCol)];
        localB[localRow][localCol] = B[(k + localRow) * N + globalCol];

        // Ensure the entire block is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute multiplication for this block
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += localA[localRow][j] * localB[j][localCol];
        }

        // Ensure computation is done before loading next block
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalCol] = sum;
}