#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char **argv)
{
	Simulator simulator;
	bool curr = simulator.run();

	return 0;
}