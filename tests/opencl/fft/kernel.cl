#include "common.h"

__kernel void fft_radix4(__global float2* input, __global float2* output, const unsigned int N) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);

    // Allocate local memory to store intermediate results and twiddle factors
    __local float2 localData[LOCAL_SIZE];
    __local float2 twiddleFactors[LOCAL_SIZE / 4];

    // Calculate twiddle factors for this FFT stage and store in local memory
    if (localId < LOCAL_SIZE / 4) {
        float angle = -2 * M_PI * localId / LOCAL_SIZE;
        twiddleFactors[localId] = (float2)(cos(angle), sin(angle));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the offset for the data this work-group will process
    int offset = groupId * LOCAL_SIZE;

    // Load a chunk of input into local memory for faster access
    if (globalId < N) {
        localData[localId] = input[globalId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the Radix-4 FFT on the data chunk in local memory
    for (unsigned int stride = 1; stride < LOCAL_SIZE; stride *= 4) {
        int twiddleIndex = (localId / stride) % 4;
        float2 twiddle = twiddleFactors[twiddleIndex * (LOCAL_SIZE / (4 * stride))];

        // Load data
        float2 data0 = localData[localId];
        float2 data1 = localData[localId + stride];
        float2 data2 = localData[localId + 2 * stride];
        float2 data3 = localData[localId + 3 * stride];

        // Apply twiddle factors
        data1 *= twiddle;
        data2 *= twiddle * twiddle;
        data3 *= twiddle * twiddle * twiddle;

        // Radix-4 butterfly operations
        float2 t0 = data0 + data2;
        float2 t1 = data0 - data2;
        float2 t2 = data1 + data3;
        float2 t3 = (data1 - data3) * (float2)(0, -1);

        // Store results
        localData[localId] = t0 + t2;
        localData[localId + stride] = t1 + t3;
        localData[localId + 2 * stride] = t0 - t2;
        localData[localId + 3 * stride] = t1 - t3;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the results back to global memory
    if (globalId < N) {
        output[globalId] = localData[localId];
    }
}
