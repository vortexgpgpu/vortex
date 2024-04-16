__kernel void gl_viewport_division(
	__global float4 *gl_position,
	__private const int4 viewport,
	__private const float2 depth_range
) {
  int ox = viewport.x+(viewport.z/2);
  int oy = viewport.y+(viewport.w/2);

  gl_position->x = viewport.z/2 * gl_position->x + ox; // z == width
  gl_position->y = viewport.w/2 * gl_position->y + oy; // w == height
  gl_position->z = (depth_range.y-depth_range.x)/2 * gl_position->z + (depth_range.y+depth_range.x)/2;
}