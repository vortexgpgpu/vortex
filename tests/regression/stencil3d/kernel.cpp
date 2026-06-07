#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg)
{
    auto A = reinterpret_cast<TYPE *>(arg->A_addr);
    auto B = reinterpret_cast<TYPE *>(arg->B_addr);
    auto size = arg->size; // Assuming 'size' now represents one dimension of a cubic space.

    // Calculate global column, row, and depth indices using both block and thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    TYPE sum = 0;
    int count = 0;

    // Stencil kernel size is assumed to be 3x3x3
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                // Compute the neighbor's index, handling boundary conditions manually
                int nz = dep + dz;
                int ny = row + dy;
                int nx = col + dx;

                // Clamp the indices to be within the boundary of the array
                if (nz < 0) {nz = 0;}
                else if (nz >= size){
                    nz = size - 1;}
                if (ny < 0) {
                    ny = 0; }
                else if (ny >= size){
                    ny = size - 1;}
                if (nx < 0) {
                    nx = 0;}
                else if (nx >= size){
                    nx = size - 1;}

                // Add the neighbor's value to sum
                sum += A[nz * size * size + ny * size + nx];
                count++;
            }
        }
    }

    // Compute the average of the sum of neighbors and write to the output array
    B[dep * size * size + row * size + col] = sum / count;
}

int main()
{
    auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65504);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x1000);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 1);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 0x1800);
        csr_write(VX_CSR_DFV_FILL_BANK_MASK, 0xFFFF);
        csr_write(VX_CSR_DFV_DCACHE_CORE_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 1);
    }
    int __ret = vx_spawn_threads(3, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 0);
    }
    return __ret;
}