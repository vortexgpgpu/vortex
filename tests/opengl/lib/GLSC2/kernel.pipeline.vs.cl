typedef struct __attribute__ ((packed)) attribute
{
	bool active;
    int size;
    __global void* mem;
};

void gl_vertex(
	__global const struct attribute* attributes,
	__global const void* uniforms,
	__global void* primitives // this pointer is actually pointing to the space were to write
);

void gl_perspective_division(
	__global float4* gl_position
);

void gl_viewport_division(
	__global float4 *gl_position,
	__private const int4 viewport,
	__private const float2 depth_range
);

__kernel void gl_main_vs(
	__global const struct attribute* attributes,
	__global const void* uniforms,
	__global void* primitives,
	__private const int4 viewport,
	__private const float2 depth_range
) {
	int gid = get_global_id(0);

	int primitive_size = 4; // always attribute 0 is refered to gl_position
	int attribute = 1;
	while(attributes[attribute].active) {
		primitive_size += attributes[attribute].size;
		++attribute;
	}

	gl_vertex(attributes, uniforms, primitives);

	gl_perspective_division(primitives + gid*primitive_size);

	gl_viewport_division(primitives + gid*primitive_size, viewport, depth_range);
}

void gl_perspective_division(
	__global float4* gl_position
) {
  int gid = get_global_id(0);
  float w = gl_position[gid+3][3];

  gl_position[gid+3][0]/=w;
  gl_position[gid+3][1]/=w;
  gl_position[gid+3][2]/=w;
}

void gl_viewport_division(
	__global float4 *gl_position,
	__private const int4 viewport,
	__private const float2 depth_range
) {
  gl_position[0] = viewport[0]/2 * gl_position[0] + viewport[2];
  gl_position[1] = viewport[1]/2 * gl_position[1] + viewport[3];
  gl_position[2] = (depth_range[1]-depth_range[0])/2 * gl_position[1] + (depth_range[1]+depth_range[0])/2;
}