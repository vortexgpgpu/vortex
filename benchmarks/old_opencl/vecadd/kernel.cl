kernel void
vecadd (__global const int *a,
	__global const int *b,
	__global int *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}