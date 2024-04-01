
__kernel void vertex (
  __global const float3 *vbo,
  __global float4 *primitives
) {
  int gid = get_global_id(0);

  primitives[gid] = (float4) (vbo[gid],1.0f);
}

__kernel void fragment (
  // user values
  __global const float4 *position,
  // implementation values 
  __global const float3 *gl_position, // position of the fragment in the window space
  __global unsigned short *gl_depth, // depth value of the fragment
  __global unsigned char *gl_stencil, // stencil value of the fragment
  __global bool *gl_discard, // if discarded
  __global float4 *gl_fragcolor // out color of the fragment || It is deprecated in OpenGL 3.0 
)
{
  int gid = get_global_id(0);

  gl_fragcolor[gid] = (float4) 1.0f;
}
