#include "lib.h"


extern void createThreads(unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *);
extern void        wspawn(unsigned, unsigned, unsigned, unsigned *, unsigned *, unsigned *);


void reschedule_warps()
{

	if (queue_isEmpty())
	{
		ECALL;
	}

	Job j;
	queue_dequeue(&j);
	asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
	createThreads(j.n_threads, j.wid, j.func_ptr, j.x, j.y, j.z);

	ECALL;

}

void schedule_warps()
{
	asm __volatile__("mv s3, sp");
	while (!queue_isEmpty() && queue_availableWarps())
	{
		++q.active_warps;
		Job j;
		queue_dequeue(&j);

		asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
		wspawn(j.n_threads, j.wid, j.func_ptr, j.x, j.y, j.z);
	}
	asm __volatile__("mv sp, s3");
}

void sleep(int t)
{
	for(int z = 0; z < t; z++) {}
}



void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{

	asm __volatile__("addi s2, sp, 0");
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
	    j.x         = x_ptr;
	    j.y         = y_ptr;
	    j.z         = z_ptr;

	    queue_enqueue(&j);
	}
	asm __volatile__("addi sp, s2, 0");


	schedule_warps();

	sleep(100);

	// asm __volatile__("addi t5, sp, 0");

	// for (unsigned i = 1; i < num_Warps; i++)
	// {
	// 	asm __volatile__("addi sp, sp, -2048");
	// 	wspawn(num_threads, i, func, x_ptr, y_ptr, z_ptr);
	// }

	// asm __volatile__("addi sp, t5, 0");

	// createThreads(num_threads, 0, (unsigned) func, x_ptr, y_ptr, z_ptr);

}


// unsigned get_wid()
// {
// 	register unsigned ret asm("s7");
// 	return ret;
// }

unsigned * get_1st_arg(void)
{
	register unsigned *ret asm("s7");
	return ret;
}
unsigned * get_2nd_arg(void)
{
	register unsigned *ret asm("s8");
	return ret;
}
unsigned * get_3rd_arg(void)
{
	register unsigned *ret asm("s9");
	return ret;
}

