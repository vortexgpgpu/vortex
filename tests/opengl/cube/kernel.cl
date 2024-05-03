
float4 mul(float16 mat, float4 vec) {
  float4 result = 0;

  for(int i=0; i<4;++i) {
    float value = 0;
    for(int j=0; j<4; j+=1) {
      value += mat[i*4+j]*vec[j];
    }
    result[i] = value;
  }

  return result;
}

__kernel void gl_main_vs (
  // my imp
  __global const float* positions,
  __global const float* color,
  const float16 perspective,
  const float16 view,
  const float16 model,
  // implementation values
  __global float4 *gl_Positions,
  __global float4 *gl_Primitives
) {
  int gid = get_global_id(0);
  // in out vars
  float x = positions[gid*3];
  float y = positions[gid*3+1];
  float z = positions[gid*3+2];

  float r = color[gid*3];
  float g = color[gid*3+1];
  float b = color[gid*3+2];

  // vertex operations
  gl_Positions[gid] =  mul(view,mul(model,(float4) (x, y, z, 1.0f)));
 
  gl_Primitives[gid*2] = (float4) (r,g,b,1.0f);
  // gl_Primitives[gid*2+1] = (float4) (1.f,1.f,1.f,1.0f);
}

__kernel void gl_main_fs (
  // user values
  const float16 perspective,
  const float16 view,
  const float16 model,
  // implementation values 
  __global float4 *gl_FragCoord, // position of the fragment in the window space, z is depth value
  __global const float4 *gl_Rasterization,
  __global bool *gl_Discard, // if discarded
  __global float4 *gl_FragColor // out color of the fragment || It is deprecated in OpenGL 3.0 
)
{
  int gid = get_global_id(0);
  // in out vars

  // fragment operations
  gl_FragColor[gid] = gl_Rasterization[gid*2];
}
