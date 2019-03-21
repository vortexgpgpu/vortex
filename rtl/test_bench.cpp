#include "Vvortex.h"
#include "verilated.h"

#include <stdio.h>

unsigned inst_array[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);

	Vvortex * vortex = new Vvortex;

	vortex->clk = 0;
	vortex->reset = 1;
	vortex->eval();

	vortex->reset = 0;

	for (int i = 0; i < 10; i++)
	{

		vortex->fe_instruction = inst_array[(vortex->curr_PC) / 4];

		vortex->clk = 1;
		vortex->eval();

		vortex->clk = 0;
		vortex->eval();


	}


	delete vortex;

	return 0;

}



