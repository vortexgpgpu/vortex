__kernel void gl_perspective_division(
	__global float4* gl_Positions
) {
  int gid = get_global_id(0);

  float w = gl_Positions[gid].w;

  gl_Positions[gid].x /= w;
  gl_Positions[gid].y /= w;
  gl_Positions[gid].z /= w;
}