#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg)
{
    auto A = reinterpret_cast<TYPE*>(arg->A_addr);
    auto B = reinterpret_cast<TYPE*>(arg->B_addr);
    auto size = arg->size;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    if (col >= (int)size || row >= (int)size || dep >= (int)size)
        return;

    TYPE sum = 0;
    int count = 0;

    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int nz = dep + dz;
                int ny = row + dy;
                int nx = col + dx;

                if (nz < 0) nz = 0; else if (nz >= (int)size) nz = size - 1;
                if (ny < 0) ny = 0; else if (ny >= (int)size) ny = size - 1;
                if (nx < 0) nx = 0; else if (nx >= (int)size) nx = size - 1;

                sum += A[nz * size * size + ny * size + nx];
                count++;
            }
        }
    }

    B[dep * size * size + row * size + col] = sum / count;
}
