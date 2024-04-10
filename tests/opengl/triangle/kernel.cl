typedef struct _attribute_pointer
{
  int type; // byte, ubyte, short, ushort, float
  int size; // to the next attribute
  __global void* mem;
} _attribute_pointer;

typedef struct _attribute_int
{
  int4 values; 
} _attribute_int;

typedef struct _attribute_float
{
  float4 values;
} _attribute_float;

typedef union _attribute {
  struct _attribute_int  attribute_int;
  struct _attribute_float  attribute_float;
  struct _attribute_pointer attribute_pointer;
} _attribute;

typedef struct __attribute__ ((packed)) attribute {
  int type;
  union _attribute attribute;
} attribute;

void gl_main_vs (
  __global const float3* position,
  // implementation values
  __global float4 *gl_Positions,
  __global float4 *gl_Primitives
) {
  int gid = get_global_id(0);
  // in out vars
  __global const float3* in_position = position + gid;
  __global float4* gl_Position = gl_Positions + gid; 

  // vertex operations
  *gl_Position = (float4) (*in_position,1.0f);
}

__kernel void gl_main_fs (
  // user values
  // implementation values 
  __global float4 *gl_FragCoord, // position of the fragment in the window space, z is depth value
  __global bool *gl_Discard, // if discarded
  __global float4 *gl_FragColor, // out color of the fragment || It is deprecated in OpenGL 3.0 
  __global const void *gl_Rasterization
)
{
  int gid = get_global_id(0);
  // in out vars

  // fragment operations
  gl_FragColor[gid] = (float4) 1.0f;
}
