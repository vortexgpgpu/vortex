__kernel void gl_viewport_division(
	__global float4 *gl_position,
	__private const int4 viewport,
	__private const float2 depth_range
) {
  gl_position->x = viewport.x/2 * gl_position->x + viewport.z; // z == width
  gl_position->y = viewport.y/2 * gl_position->y + viewport.w; // w == height
  gl_position->z = (depth_range.y-depth_range.x)/2 * gl_position->z + (depth_range.y+depth_range.x)/2;
}