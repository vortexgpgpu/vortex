#version 450
// NV (4) output control points from the input patch. The levels are the
// specialization constants (known when the pipeline is made) or the first
// input control point's attributes (known only on the device).
#ifndef NV
#define NV 4
#endif
layout(vertices = NV) out;
layout(constant_id = 0) const bool DYN = false;
layout(constant_id = 1) const float LO0 = 1.0;
layout(constant_id = 2) const float LO1 = 1.0;
layout(constant_id = 3) const float LO2 = 1.0;
layout(constant_id = 4) const float LO3 = 1.0;
layout(constant_id = 5) const float LI0 = 1.0;
layout(constant_id = 6) const float LI1 = 1.0;
layout(location = 0) in vec2 v_pos[];
layout(location = 1) in vec4 v_lo[];
layout(location = 2) in vec2 v_li[];
layout(location = 0) out vec2 t_pos[];
void main() {
  // A three-point patch makes its fourth corner from the other three.
  t_pos[gl_InvocationID] = gl_InvocationID < gl_PatchVerticesIn ? v_pos[gl_InvocationID]
                                                      : v_pos[0] + (v_pos[2] - v_pos[1]);
  if (gl_InvocationID == 0) {
    if (DYN) {
      gl_TessLevelOuter[0] = v_lo[0].x;
      gl_TessLevelOuter[1] = v_lo[0].y;
      gl_TessLevelOuter[2] = v_lo[0].z;
      gl_TessLevelOuter[3] = v_lo[0].w;
      gl_TessLevelInner[0] = v_li[0].x;
      gl_TessLevelInner[1] = v_li[0].y;
    } else {
      gl_TessLevelOuter[0] = LO0;
      gl_TessLevelOuter[1] = LO1;
      gl_TessLevelOuter[2] = LO2;
      gl_TessLevelOuter[3] = LO3;
      gl_TessLevelInner[0] = LI0;
      gl_TessLevelInner[1] = LI1;
    }
  }
}
