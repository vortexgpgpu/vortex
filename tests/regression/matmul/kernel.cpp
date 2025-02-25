#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t task_id = blockIdx.x;
	int32_t* src0_ptr = (int32_t*)arg->src0_addr;
	int32_t* src1_ptr = (int32_t*)arg->src1_addr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_addr;
	uint64_t a_addr = reinterpret_cast<uint64_t>(src0_ptr);
	uint64_t b_addr = reinterpret_cast<uint64_t>(src1_ptr);
	uint64_t c_addr = reinterpret_cast<uint64_t>(dst_ptr);

	uint32_t tc_size = arg->tc_size;
	uint32_t TC_per_warp = arg->TC_per_warp;
	unsigned num_threads = arg->num_threads;
	int num_warps = arg->num_warps;
	uint32_t matrix_size = arg->matrix_size;
	
	int n_tiles = matrix_size/tc_size;
	int num_output_tiles = (matrix_size*matrix_size)/(tc_size*tc_size);
	
	int num_tasks = arg->num_tasks;

	//Assuming matrix size always > tensor core size
	int warps_actual;
	if (TC_per_warp > num_output_tiles)
		warps_actual = 1;
	else 
		warps_actual = num_output_tiles/TC_per_warp;

	int num_warps_actual = (warps_actual < num_warps)? warps_actual: num_warps;
	int num_threads_per_tc = (1> num_threads/TC_per_warp)? 1: num_threads/TC_per_warp;
	
	int num_tasks_per_thread = (1> (num_tasks/(num_threads*num_warps_actual)))? 1: (num_tasks/(num_threads*num_warps_actual));
	int num_tasks_per_warp = (1 > num_tasks/num_warps_actual)? 1:num_tasks/num_warps_actual;
	int task_id_first_warp = task_id%num_tasks_per_warp;

	//A&B
	int num_data_per_op_tile = tc_size*tc_size*n_tiles;
	int num_data_per_warp = num_data_per_op_tile*((1> (num_output_tiles/num_warps_actual))?1:(num_output_tiles/num_warps_actual));
	
	int addr_shift;
	if (((tc_size*tc_size*n_tiles)/(num_threads)) > 1)
		addr_shift = (tc_size*tc_size*n_tiles)/(num_threads);
	else
		addr_shift = 1;
	//Offset for 1st warp
	int offset = ((task_id_first_warp/num_tasks_per_thread)*addr_shift) + ((task_id_first_warp%num_tasks_per_thread)*num_data_per_op_tile);
	offset = offset + (num_data_per_warp*(task_id/num_tasks_per_warp));

	//C
	int num_data_per_op_tile_c = tc_size*tc_size;
	int num_data_per_warp_c = num_data_per_warp/n_tiles;
	
	int addr_shift_c;
	if (((tc_size*tc_size)/(num_threads)) > 1)
		addr_shift_c = tc_size;
	else
		addr_shift_c = 1;
	//Offset for 1st warp
	int offset_c = ((task_id_first_warp/num_tasks_per_thread)*addr_shift_c) + ((task_id_first_warp%num_tasks_per_thread)*num_data_per_op_tile_c);
	offset_c = offset_c + (num_data_per_warp_c*(task_id/num_tasks_per_warp));
	
	int thread_limit = (num_threads < tc_size*tc_size*n_tiles*TC_per_warp)? num_threads : tc_size*tc_size*n_tiles*TC_per_warp;
	int thread_limit_c = (num_threads<tc_size*tc_size)? num_threads:tc_size*tc_size;
	
	//OLD TASK DISTRIBUTION // For 8x8 matrix, 2x2 tc_size, 1 tc_num, 4threads, 2warps => 64 tasks => 32 tasks/warp => 8 tasks/thread
	/*task0->thread0, warp0
	task1->thread0 , warp0
	task2->thread0 , warp0
	.
	task7->thread0 
	task8->thread1
	task9->thread1 
	.
	.
	------
	task32 -> thread0, warp1
	task33 -> thread1, warp1
	.
	*/

	//NEW TASK DISTRIBUTION // For 8x8 matrix, 2x2 tc_size, 1 tc_num, 4threads, 2warps => 64 tasks => 32 tasks/warp => 8 tasks/thread
	/*task0->thread0, warp0
	task1->thread1 , warp0
	task2->thread2 , warp0
	task3->thread3 ,...
	task4->thread0
	task5->thread1 
	.
	.
	------
	task32 -> thread0, warp1
	task33 -> thread1, warp1
	.
	.*/
	
	//TODO :: change this for new task->thread distribution
	if (((task_id%num_tasks_per_warp)/num_tasks_per_thread) < thread_limit)
	{	
		uint64_t a_addr_base = a_addr + offset*arg->data_size;
		uint64_t b_addr_base = b_addr + offset*arg->data_size;
		uint64_t c_addr_base = c_addr + offset_c*arg->data_size;
		csr_write(VX_MAT_MUL_SIZE,n_tiles);
		csr_write(VX_TC_NUM,TC_per_warp);
		csr_write(VX_TC_SIZE,tc_size);

		vx_matrix_load (0, a_addr_base);
		vx_matrix_load (1, b_addr_base);
		//In case of multiple threads - sync load
		vx_fence();

		vx_matrix_mul();   //Assuming padding to ensure matrix size is a multiple of tc_size
		vx_fence();
		if (((task_id%num_tasks_per_warp)/num_tasks_per_thread) < thread_limit_c)
			vx_matrix_store(c_addr_base);
		//In case of multiple threads - sync store
		vx_fence();
	}	
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
