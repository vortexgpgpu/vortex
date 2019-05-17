
#include "vx_mem.h"


void * vx_malloc_shared(unsigned size)
{
	void * to_return;

	bool done = false;
	unsigned curr_size;

	unsigned curr_index = 0;
	while ((curr_index < free_index) && !done)
	{
		curr_size = (unsigned) *(free_array[curr_index].ptr - 4);
		if (curr_size <= size)
		{
			to_return = free_array[curr_index].ptr;
			done = true;
		}

		curr_index++;
	}

	unsigned * u_heap_ptr = (unsigned *) heap_ptr;

	if (!done)
	{
		u_heap_ptr[0] = size;
		to_return     = heap_ptr + 4;
		heap_ptr      = to_return + size;
	}

	return to_return;
}

void vx_free(void * to_free)
{

}