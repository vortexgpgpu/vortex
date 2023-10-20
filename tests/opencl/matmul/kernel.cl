__kernel void matmul(__global float *A, 
                     __global float *B, 
                     __global float *C, 
                     const unsigned int N, 
                     __local float *localA, 
                     __local float *localB)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);
    int localSize = get_local_size(0);  // assuming square local size

    float sum = 0.0f;

    // Loop over all blocks of both matrices
    for (int k = 0; k < N; k += localSize) {
        // Load block of matrix A to local memory
        localA[localRow * localSize + localCol] = A[row * N + k + localCol];

        // Load block of matrix B to local memory, adjusting for column-major access
        localB[localRow * localSize + localCol] = B[(k + localRow) * N + col];

        // Synchronize to make sure the tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrix blocks and accumulate result
        for (int j = 0; j < localSize; j++) {
            sum += localA[localRow * localSize + j] * localB[j * localSize + localCol];
        }

        // Synchronize before loading the next block
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row * N + col] = sum;
}

/*__kernel void matmul(__global float *A, __global float *B, __global float *C, const unsigned int N)
{
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    // Static local memory declaration
    __local float localA[16][16];
    __local float localB[16][16];

    float sum = 0.0f;

    // Iterate over blocks
    for (int k = 0; k < N; k += 16) {
        // Load a block of matrix A into local memory
        localA[localRow][localCol] = A[globalRow * N + k + localCol];

        // Load a block of matrix B into local memory
        localB[localRow][localCol] = B[(k + localRow) * N + globalCol];

        // Ensure the entire block is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute multiplication for this block
        for (int j = 0; j < 16; j++) {
            sum += localA[localRow][j] * localB[j][localCol];
        }

        // Wait until all threads have computed before loading the next block
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalCol] = sum;
}*/