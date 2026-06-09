__kernel void parallelSum(__global const float* input, __global float* output, int N, __local float* temp) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);

    // Load input into local memory
    if (global_id < N) {
        temp[local_id] = input[global_id];
    } else {
        temp[local_id] = 0.0f;  // Pad with zero for out-of-range elements
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (unsigned int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            temp[local_id] += temp[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result of this local reduction to global memory
    if (local_id == 0) {
        output[get_group_id(0)] = temp[0];
    }
}
