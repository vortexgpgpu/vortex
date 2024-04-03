
void gl_vertex (
  const float3 *attributes,
  const void *uniforms, 
  float4 *primitives
) {
  int gid = get_global_id(0);

  primitives[gid] = (float4) (attributes[0],1.0f);
}

__kernel void gl_fragment (
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
