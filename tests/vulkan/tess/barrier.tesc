#version 450
// Three rounds of every invocation reading its neighbour's value and, after a
// barrier, adding half of it to its own, a barrier between rounds: the values
// pass.tese colours by. With NESTED the rounds are a loop whose count the
// compiler cannot know, so its barriers sit in control flow.
layout(vertices = 4) out;
layout(location = 0) in vec2 v_pos[];
layout(location = 1) in vec4 v_lo[];
layout(location = 2) in vec2 v_li[];
layout(location = 0) out vec2 t_pos[];
layout(location = 1) out float t_val[];
layout(location = 2) out float t_next[];
layout(location = 3) patch out float p_sum;
// GLSL keeps a control stage's barrier() out of control flow, and SPIR-V does
// not: barrier_nested.spvasm is this shader built with NESTED, its
// memoryBarrierImage() placeholders then made the OpControlBarrier barrier()
// is.
#ifdef NESTED
#define ROUND_BARRIER() memoryBarrierImage()
#else
#define ROUND_BARRIER() barrier()
#endif
#define ROUND()                                         \
  {                                                       \
    const float n = t_val[(gl_InvocationID + 1) & 3];     \
    ROUND_BARRIER();                                      \
    t_val[gl_InvocationID] += n * 0.5;                    \
    ROUND_BARRIER();                                      \
  }
void main() {
  t_pos[gl_InvocationID] = v_pos[gl_InvocationID];
  t_val[gl_InvocationID] = v_pos[gl_InvocationID].x * 2.0 + float(gl_InvocationID) * 0.125;
  barrier();
#ifdef NESTED
  const int rounds = v_pos[0].y >= -2.0 ? 3 : 1;
  for (int r = 0; r < rounds; r++)
    ROUND()
#else
  ROUND()
  ROUND()
  ROUND()
#endif
  t_next[gl_InvocationID] = t_val[(gl_InvocationID + 1) & 3];
  if (gl_InvocationID == 0) {
    p_sum = t_val[0] + t_val[1] + t_val[2] + t_val[3];
    gl_TessLevelOuter[0] = v_lo[0].x;
    gl_TessLevelOuter[1] = v_lo[0].y;
    gl_TessLevelOuter[2] = v_lo[0].z;
    gl_TessLevelOuter[3] = v_lo[0].w;
    gl_TessLevelInner[0] = v_li[0].x;
    gl_TessLevelInner[1] = v_li[0].y;
  }
}
