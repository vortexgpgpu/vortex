#include "lib.h"

// namespace Sphinx
// {


void createThreads(unsigned num_threads, unsigned wid, unsigned func_addr, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{
	register unsigned *xx  asm("s2")  = x_ptr;
	register unsigned *yy  asm("s3")  = y_ptr;
	register unsigned *zz  asm("s4")  = z_ptr;
	register unsigned wid_ asm("a1")  = wid;

	asm __volatile__("addi t5, sp, 0");
	for (unsigned i = 1; i < num_threads; i++)
	{

		register unsigned cur_tid asm("a0") = i;
		register unsigned not_sure asm("t1") = i;
		asm __volatile__("addi sp, sp, -256");
		CLONE;
	}
	asm __volatile__("addi sp, t5, 0");


	register unsigned cur_tid asm("a0") = 0;


	// jalis TO FUNC
	register unsigned num_lanes asm("t6")  = func_addr;
	register unsigned link      asm("s11") = num_threads;


	JALRS;
	ECALL;

}

void wspawn(unsigned num_threads, unsigned wid, FUNC, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{


	register unsigned *tzz  asm("t2") = z_ptr;

	register unsigned func_add  asm("t1") = (unsigned) &createThreads;

	register unsigned n_threads asm("a0") = num_threads;
	register unsigned wwid   asm("a1") = wid;
	register unsigned ffunc  asm("a2") = (unsigned) func;

	register unsigned *xx  asm("a3") = x_ptr;
	register unsigned *yy  asm("a4") = y_ptr;
	register unsigned *zz  asm("a5") = tzz;

	WSPAWN; // THIS SHOULD COPY THE CSR REGISTERS TO THE NEW WARP

}



void createWarps(unsigned num_Warps, unsigned num_threads, FUNC, unsigned * x_ptr, unsigned * y_ptr, unsigned * z_ptr)
{
	asm __volatile__("addi t5, sp, 0");

	for (unsigned i = 1; i < num_Warps; i++)
	{
		asm __volatile__("addi sp, sp, -2048");
		wspawn(num_threads, i, func, x_ptr, y_ptr, z_ptr);
	}

	asm __volatile__("addi sp, t5, 0");

	createThreads(num_threads, 0, (unsigned) func, x_ptr, y_ptr, z_ptr);


	ECALL;

}


unsigned get_wid()
{
	register unsigned ret asm("s1");
	return ret;
}

unsigned * get_1st_arg(void)
{
	register unsigned *ret asm("s2");
	return ret;
}
unsigned * get_2nd_arg(void)
{
	register unsigned *ret asm("s3");
	return ret;
}
unsigned * get_3rd_arg(void)
{
	register unsigned *ret asm("s4");
	return ret;
}

void initiate_stack()
{
	asm __volatile__("lui  sp,0x7ffff");
}
