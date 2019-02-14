#include "lib.h"

// namespace Sphinx
// {

#define FUNC void (func)(unsigned)



void set_wid(unsigned i)
{
	SET_WID(i);
}

void set_func(FUNC)
{
	SET_FUNC(func);
}

unsigned get_func()
{
	unsigned ret;
	GET_FUNC(ret);
	return ret;
}

unsigned get_wid()
{
	unsigned ret;
	GET_WID(ret);
	return ret;
}

void createThreads(unsigned num_threads, unsigned wid, unsigned func_addr)
{

	asm __volatile__("addi t5, sp, 0");
	for (unsigned i = 1; i < num_threads; i++)
	{

		register unsigned cur_tid asm("t1") = i;
		asm __volatile__("addi sp, sp, -256");
		CLONE;
	}
	asm __volatile__("addi sp, t5, 0");


	register unsigned cur_tid asm("t1") = 0;


	// jalis TO FUNC
	register unsigned num_lanes asm("t6")  = func_addr;
	register unsigned link      asm("s11") = num_threads;

	register unsigned n_threads asm("a0")  = wid;
	JALRS;
	ECALL;

}

void wspawn(unsigned num_threads, unsigned wid, FUNC)
{

	// set_wid(wid);
	// set_func(func);

	

	register unsigned n_threads asm("a0") = num_threads;
	register unsigned wwid   asm("a1") = wid;
	register unsigned ffunc  asm("a2") = (unsigned) func;

	register unsigned func_add  asm("t1") = (unsigned) &createThreads;
	WSPAWN; // THIS SHOULD COPY THE CSR REGISTERS TO THE NEW WARP

}



void createWarps(unsigned num_Warps, unsigned num_threads, FUNC)
{
	// asm __volatile__("addi t5, sp, 0");

	// for (unsigned i = 1; i < num_Warps; i++)
	// {
	// 	asm __volatile__("addi sp, sp, -2048");
	// 	wspawn(num_threads, i, func);
	// }

	// asm __volatile__("addi sp, t5, 0");

	createThreads(num_threads, 0, (unsigned) func);


	ECALL;

}


unsigned get_tid()
{
	register unsigned tid asm("t1");
}

void initiate_stack()
{
	asm __volatile__("lui  sp,0x7ffff");
}