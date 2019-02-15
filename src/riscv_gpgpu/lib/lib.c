#include "lib.h"

void createThreads(unsigned num_threads, unsigned wid, unsigned func_addr, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{

	asm __volatile__("mv t6, a0");
	asm __volatile__("mv a1, a1");

	asm __volatile__("mv s7, a3");
	asm __volatile__("mv s8, a4");
	asm __volatile__("mv s9, a5");


	asm __volatile__("addi t5, sp, 0");

	register unsigned num_threads_ asm("t6");
	for (unsigned i = 1; i < num_threads_; i++)
	{

		register unsigned cur_tid asm("a0") = i;
		register unsigned not_sure asm("t1") = i;
		asm __volatile__("addi sp, sp, -2048");
		CLONE;
	}
	asm __volatile__("addi sp, t5, 0");


	register unsigned cur_tid asm("a0") = 0;


	// jalis TO FUNC
	register unsigned num_lanes asm("t6")  = func_addr;
	register unsigned link      asm("s11") = num_threads;


	JALRS;

	register unsigned jump_dest asm("a0") = (unsigned) reschedule_warps;
	JMPRT;



	// // register unsigned *xx  asm("s7")  = x_ptr;
	// // register unsigned *yy  asm("s8")  = y_ptr;
	// // register unsigned *zz  asm("s9")  = z_ptr;
	// register unsigned wid_ asm("a1")  = wid;


	// asm __volatile__("addi t5, sp, 0");
	// for (unsigned i = 1; i < num_threads; i++)
	// {

	// 	register unsigned cur_tid asm("a0") = i;
	// 	register unsigned not_sure asm("t1") = i;
	// 	asm __volatile__("addi sp, sp, -256");
	// 	CLONE;
	// }
	// asm __volatile__("addi sp, t5, 0");


	// register unsigned cur_tid asm("a0") = 0;


	// // jalis TO FUNC
	// register unsigned num_lanes asm("t6")  = func_addr;
	// register unsigned link      asm("s11") = num_threads;


	// JALRS;

	// register unsigned jump_dest asm("a0") = (unsigned) reschedule_warps;
	// JMPRT;

}

void wspawn(unsigned num_threads, unsigned wid, unsigned func, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{

	asm __volatile__("mv t2, a5");
	// asm __volatile__("mv t1, a5");

	register unsigned func_add  asm("t1") = (unsigned) &createThreads;



	WSPAWN; // THIS SHOULD COPY THE CSR REGISTERS TO THE NEW WARP



	// register unsigned *tzz  asm("t2") = z_ptr;

	// register unsigned func_add  asm("t1") = (unsigned) &createThreads;

	// register unsigned n_threads asm("a0") = num_threads;
	// register unsigned wwid   asm("a1")    = wid;
	// register unsigned ffunc  asm("a2")    = func;

	// register unsigned *xx  asm("a3") = x_ptr;
	// register unsigned *yy  asm("a4") = y_ptr;
	// register unsigned *zz  asm("a5") = tzz;

	// WSPAWN; // THIS SHOULD COPY THE CSR REGISTERS TO THE NEW WARP

}

void reschedule_warps()
{

	if (queue_isEmpty())
	{
		ECALL;
	}

	Job j;
	queue_dequeue(&j);
	asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
	wspawn(j.n_threads, j.wid, j.func_ptr, j.x, j.y, j.z);

}

void schedule_warps()
{
	asm __volatile__("mv t5, sp");
	while (!queue_isEmpty() && queue_availableWarps())
	{
		Job j;
		queue_dequeue(&j);

		asm __volatile__("mv sp,%0"::"r" (j.base_sp):);
		wspawn(j.n_threads, j.wid, j.func_ptr, j.x, j.y, j.z);
	}
	asm __volatile__("mv sp, t5");
}

void sleep()
{
	for(int z = 0; z < 10000; z++) {}
}



void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{

	asm __volatile__("addi t5, sp, 0");
	for (unsigned i = 0; i < num_Warps; i++)
	{
		asm __volatile__("lui t6, 0xFFFF0");
		asm __volatile__("add sp, sp, t6");
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
	asm __volatile__("addi sp, t5, 0");


	schedule_warps();

	sleep();

	// asm __volatile__("addi t5, sp, 0");

	// for (unsigned i = 1; i < num_Warps; i++)
	// {
	// 	asm __volatile__("addi sp, sp, -2048");
	// 	wspawn(num_threads, i, func, x_ptr, y_ptr, z_ptr);
	// }

	// asm __volatile__("addi sp, t5, 0");

	// createThreads(num_threads, 0, (unsigned) func, x_ptr, y_ptr, z_ptr);

}


unsigned get_wid()
{
	register unsigned ret asm("s7");
	return ret;
}

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

void initiate_stack()
{
	asm __volatile__("lui  sp,0x7ffff":::);
}
