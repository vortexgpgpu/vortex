

#include "d_cache_test_bench.h"

//#define NUM_TESTS 46

int main(int argc, char **argv)
{

	Verilated::commandArgs(argc, argv);

	Verilated::traceEverOn(true);


	VX_d_cache v;


	bool curr = v.simulate();
	//if ( curr) std::cerr << GREEN << "Test Passed: " << testing << std::endl;
	//if (!curr) std::cerr << RED   << "Test Failed: " << testing << std::endl;
	if ( curr) std::cerr << GREEN << "Test Passed: " << std::endl;
	if (!curr) std::cerr << RED   << "Test Failed: " << std::endl;

	return 0;

}



