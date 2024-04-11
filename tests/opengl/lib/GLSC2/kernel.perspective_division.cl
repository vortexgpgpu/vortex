__kernel void gl_perspective_division(
	__global float4* gl_position
) {
  int gid = get_global_id(0);
  float w = gl_position[gid+3][3];

  gl_position[gid+3][0]/=w;
  gl_position[gid+3][1]/=w;
  gl_position[gid+3][2]/=w;
}