
#include "vx_io.h"

void vx_print_hex(unsigned f)
{
	vx_print_str(hextoa[f]);
}


void vx_printf(char * c, unsigned f)
{
	vx_print_str(c);
	vx_print_hex(f);
	vx_print_str("\n");
}