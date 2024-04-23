/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
//#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

#include "gpu_info.h"

void compute_active_thread(size_t *thread,
			   size_t *grid,
			   int task,
			   int pad,
			   int major,
			   int minor,
			   int sm)
{
	int max_thread;
	int max_block=8;
	if(major==1)
	{
		if(minor>=2)
			max_thread=1024;
		else
			max_thread=768;
	}
	else if(major==2)
		max_thread=1536;
	else
		//newer GPU  //keep using 2.0
		max_thread=1536;
	
	int _grid;
	int _thread;
	
	if(task*pad>sm*max_thread)
	{
		_thread=max_thread/max_block;
		_grid = ((task*pad+_thread-1)/_thread)*_thread;
	}
	else
	{
		_thread=pad;
		_grid=task*pad;
	}

	thread[0]=_thread;
	grid[0]=_grid;
}
