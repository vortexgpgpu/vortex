
#include "vx_back.h"
#include "../vx_io/vx_io.h"


void vx_before_main()
{
	// unsigned num_available_warps = vx_available_warps();
	for (int i = 0; i < 8; i++)
	{
		queue_initialize(q + i);
	}
}

void vx_reschedule_warps()
{


	register unsigned curr_warp asm("s10");
	// vx_printf("Reschedule: ", curr_warp);

	if (queue_isEmpty(q+curr_warp))
	{
		// vx_printf("Done: ", curr_warp);
		done[curr_warp] = 1;
		if (curr_warp == 0)
		{
			vx_load_context();
			return;
		}
		ECALL;
	}

	Job j;
	queue_dequeue(q+curr_warp,&j);

	// vx_printf("Reschedule -> ", j.wid);
	asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
	vx_createThreads(j.n_threads, j.wid, j.func_ptr, j.args, j.assigned_warp);

	ECALL;

}

void vx_schedule_warps()
{

	unsigned num_available_warps = vx_available_warps();

	asm __volatile__("mv s3, sp");

	for (int curr_warp = 1; curr_warp < num_available_warps; ++curr_warp)
	{
		if (!queue_isEmpty(q+curr_warp)) 
		{
			Job j;
			queue_dequeue(q+curr_warp,&j);
			asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
			vx_wspawn(j.n_threads, j.wid, j.func_ptr, j.args, j.assigned_warp);
		}
	}

	asm __volatile__("mv sp, s3");


	vx_save_context();

	// vx_print_str("saved context\n");

	register unsigned val asm("tp");
	if (val)
	{
		if (!queue_isEmpty(q))
		{
			// vx_print_str("found something for w0\n");
			Job j;
			queue_dequeue(q,&j);
			// vx_printf("num_threads: ", j.n_threads);
			asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
			vx_createThreads(j.n_threads, j.wid, j.func_ptr, j.args, j.assigned_warp);
		}
	}

}


void vx_spawnWarps(unsigned num_Warps, unsigned num_threads, FUNC, void * args)
{
	vx_before_main();

	unsigned num_available_warps = vx_available_warps();
	// vx_printf("Num available warps: ", num_available_warps);

	asm __volatile__("addi s2, sp, 0");
	int warp = 0;
	for (unsigned i = 0; i < num_Warps; i++)
	{
		asm __volatile__("lui s3, 0xFFFF0");
		asm __volatile__("add sp, sp, s3");
		register unsigned stack_ptr asm("sp");

		Job j;
		j.wid       = i;
		j.n_threads = num_threads;
		j.base_sp   = stack_ptr;
	    j.func_ptr  = (unsigned) func;
	    j.args      = args;
	    j.assigned_warp = warp;

	    queue_enqueue(q + warp,&j);
	    ++warp;
	    if (warp >= num_available_warps) warp = 0;
	}
	asm __volatile__("addi sp, s2, 0");


	vx_schedule_warps();

}

void vx_wait_for_warps(unsigned num_wait)
{
	// vx_printf("wait for: ", num_wait);
	unsigned num_available_warps = vx_available_warps();
	unsigned num = 0;
	while (num != num_wait)
	{
		num = 0;
		for (int i = 0; i < num_available_warps; i++)
		{
			if (done[i] == 1)
			{
				num += 1;
			}
		}
	}

	// vx_printf("num found: ", num);
	for (int i = 0; i < num_available_warps; i++) done[i] = 0;
}


void * vx_get_arg_struct(void)
{
	register void *ret asm("s7");
	return ret;
}



