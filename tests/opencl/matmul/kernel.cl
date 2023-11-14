__kernel void matmul(__global float *A, 
                     __global float *B, 
                     __global float *C, 
                     const unsigned int N, 
                     __local float *localA, 
                     __local float *localB)
{
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);
    int localSize = get_local_size(0);  // assuming square local size

    float sum = 0.0f;

    // Load initial blocks of A and B into local memory
    int k = 0;
    localA[localRow * localSize + localCol] = A[globalRow * N + k + localCol];
    localB[localRow * localSize + localCol] = B[(k + localRow) * N + globalCol];

    // Iterate over blocks
    for (k = 0; k < N; k += 16) {
        // Ensure the initial block is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute multiplication for this block
        for (int j = 0; j < 16; j++) {
            sum += localA[localRow * localSize + j] * localB[j * localSize + localCol];
        }

        // Load the next block of matrix A into local memory
        if (k + 16 < N) {
            localA[localRow * localSize + localCol] = A[globalRow * N + k + 16 + localCol];
            localB[localRow * localSize + localCol] = B[(k + 16 + localRow) * N + globalCol];
        }
    }

    C[globalRow * N + globalCol] = sum;
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

    // Load initial blocks of A and B into local memory
    int k = 0;
    localA[localRow][localCol] = A[globalRow * N + k + localCol];
    localB[localRow][localCol] = B[(k + localRow) * N + globalCol];

    // Iterate over blocks
    for (k = 0; k < N; k += 16) {
        // Ensure the initial block is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute multiplication for this block
        for (int j = 0; j < 16; j++) {
            sum += localA[localRow][j] * localB[j][localCol];
        }

        // Load the next block of matrix A into local memory
        if (k + 16 < N) {
            localA[localRow][localCol] = A[globalRow * N + k + 16 + localCol];
            localB[localRow][localCol] = B[(k + 16 + localRow) * N + globalCol];
        }
    }

    C[globalRow * N + globalCol] = sum;
}*/