#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"

void kernel_body(int local_task_id, int group_id, int local_group_id, int warps_per_group, kernel_arg_t *arg) {
	auto local_ptr = reinterpret_cast<TYPE*>(arg->local_addr);
	auto A_ptr     = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B_ptr     = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C_ptr     = reinterpret_cast<TYPE*>(arg->C_addr);
	auto size      = arg->size;
  auto tile_size = arg->tile_size;
	auto num_groups = arg->num_groups;
	auto group_size = arg->group_size;
	auto num_tiles = size / tile_size;

	// Determine row and column indices of the current subtask
	auto l_row = local_task_id / tile_size;
	auto l_col = local_task_id % tile_size;

	// Determine row and column indices of the current task
	auto g_row = (group_id / num_tiles) * tile_size + l_row;
  auto g_col = (group_id % num_tiles) * tile_size + l_col;

	// Allocate local memory for the tile of matrix A & B
	auto local_A = local_ptr + local_group_id * group_size * 2;
	auto local_B = local_A + group_size;

	TYPE sum(0);

	// Loop over tiles
	for (uint32_t k = 0; k < size; k += tile_size) {
		// Load tile of matrix A & B to local memory
		local_A[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
		local_B[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];

		// Synchronize all warps in current group
		vx_barrier(local_group_id * 2 + 0, warps_per_group);

		// Compute partial sum for the local tile
		for (uint32_t j = 0; j < tile_size; ++j) {
			sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
		}

		// Synchronize all warps in current group
		vx_barrier(local_group_id * 2 + 1, warps_per_group);
	}

	// Store the computed sum into the result matrix C
	C_ptr[g_row * size + g_col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	vx_spawn_task_groups(arg->num_groups, arg->group_size, (vx_spawn_task_groups_cb)kernel_body, arg);
	return 0;
}
