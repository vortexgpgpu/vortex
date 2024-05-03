__kernel void conv3(__global float* output, 
                    __global float* input,
                    __global float* weights, 
                    const int width, 
                    const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Adjust for padded borders
    int paddedWidth = width + 2;
    int paddedX = x + 1;
    int paddedY = y + 1;

    // Compute 3x3 convolution sum
    float sum = 0.0f;

    sum += input[(paddedY - 1) * paddedWidth + (paddedX - 1)] * weights[0]; // Top-left
    sum += input[(paddedY - 1) * paddedWidth + paddedX] * weights[1];       // Top-center
    sum += input[(paddedY - 1) * paddedWidth + (paddedX + 1)] * weights[2]; // Top-right

    sum += input[paddedY * paddedWidth + (paddedX - 1)] * weights[3];       // Middle-left
    sum += input[paddedY * paddedWidth + paddedX] * weights[4];             // Center
    sum += input[paddedY * paddedWidth + (paddedX + 1)] * weights[5];       // Middle-right

    sum += input[(paddedY + 1) * paddedWidth + (paddedX - 1)] * weights[6]; // Bottom-left
    sum += input[(paddedY + 1) * paddedWidth + paddedX] * weights[7];       // Bottom-center
    sum += input[(paddedY + 1) * paddedWidth + (paddedX + 1)] * weights[8]; // Bottom-right

    // Store the result in the output array
    output[y * width + x] = sum;
}