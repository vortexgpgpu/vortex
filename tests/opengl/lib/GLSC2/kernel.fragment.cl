

__kernel void vertex_shader (
  __global unsigned int first,
  __global const void* vao,
  __global void* primitives,
  __global const void* vbo
) {
  int gid = get_global_id(0);
  vec3f position;
  vertexatrib3f(0, gid, &position.x, &position.y, &position.z, (VAO*) vao, (float*) vbo);

  vec4* gl_position = ((float*) primitives)[4*gid];

  gl_position.x = position.x;
  gl_position.y = position.y;
  gl_position.z = position.z;
  gl_position.w = 1;
}
