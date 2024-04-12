__kernel void gl_fill(
  const unsigned int pattern,
	__global unsigned int *mem
) {
  int gid = get_global_id(0);

  mem[gid] = pattern;
}
