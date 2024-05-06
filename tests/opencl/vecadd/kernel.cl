
#include "common.h"

__kernel void vecadd (__global const TYPE *A,
	                    __global const TYPE *B,
	                    __global TYPE *C)
{
  int gid = get_global_id(0);
  C[gid] = A[gid] + B[gid];
}