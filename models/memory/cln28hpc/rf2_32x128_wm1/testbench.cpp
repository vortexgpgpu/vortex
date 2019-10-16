

#include "Vrf2_32x128_wm1_rtl.h"
#include "verilated.h"

int main()
{
	Vrf2_32x128_wm1_rtl module;

	for (int i = 0; i < 10; i++)
	{
		// module.clk = 0;
    	module.eval();
        // module.clk = 1;
        module.eval();
	}

	return 0;
}