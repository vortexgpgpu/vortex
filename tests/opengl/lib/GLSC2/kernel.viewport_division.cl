__kernel void gl_viewport_division(
	__global float4 *gl_position,
	__private const int4 viewport,
	__private const float2 depth_range
) {
  gl_position[0] = viewport[0]/2 * gl_position[0] + viewport[2];
  gl_position[1] = viewport[1]/2 * gl_position[1] + viewport[3];
  gl_position[2] = (depth_range[1]-depth_range[0])/2 * gl_position[1] + (depth_range[1]+depth_range[0])/2;
}