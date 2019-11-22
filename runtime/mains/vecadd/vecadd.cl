
#include "../../intrinsics/vx_intrinsics.h"

kernel void
vecadd (__global const int *a,
	__global const int *b,
	__global int *c)
{
  int gid = get_global_id(0);

  __if (gid < 2)
  {
  	c[gid] = a[gid] + b[gid];
  }
  __else
  {
  	c[gid] = b[gid] - a[gid];
  }
  __endif
}
