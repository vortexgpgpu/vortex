__kernel void gl_perspective_division(
	__global float4* gl_position
) {
  int gid = get_global_id(0);

  float w = gl_position[gid].w;

  gl_position[gid].x/=w;
  gl_position[gid].y/=w;
  gl_position[gid].z/=w;
}