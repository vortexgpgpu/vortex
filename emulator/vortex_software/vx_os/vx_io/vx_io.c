
#include "vx_io.h"

void vx_print_hex(unsigned f)
{
	if (f < 16)
	{
		vx_print_str(hextoa[f]);
		return;
	}
	int temp;
	int sf = 32;
	bool start = false;
	do
	{
		temp = (f >> (sf - 4)) & 0xf;
		if (temp != 0) start = true;
		if (start) vx_print_str(hextoa[temp]);
		sf -= 4;
	} while(sf > 0);
}


void vx_printf(char * c, unsigned f)
{
	vx_print_str(c);
	vx_print_hex(f);
	vx_print_str("\n");
}