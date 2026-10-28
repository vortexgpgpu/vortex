#version 450
#ifdef VIEW
#extension GL_EXT_multiview : require
#endif
// The domain point mapped into its patch's cell, coloured by gl_TessCoord, or
// by gl_PrimitiveID with ID set; WIDE adds outputs no later stage reads, VIEW
// moves view 1's patches half the target right. Every operation is exact: the
// cell corners are multiples of 1/64, the coordinates multiples of 2^-16.
layout(DOMAIN, SPACING, WINDING POINTS) in;
layout(location = 0) in vec2 t_pos[];
#ifdef XFB
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) out vec4 color;
#else
layout(location = 0) out vec4 color;
#endif
#ifdef WIDE
layout(location = 1) out vec4 wide[7];
#endif
void main() {
  const vec2 o = t_pos[0];
  const vec2 dx = t_pos[1] - o, dy = t_pos[3] - o;
  const vec3 tc = gl_TessCoord;
  gl_Position = vec4(o + tc.x * dx + tc.y * dy, 0.0, 1.0);
#ifdef VIEW
  // view 1 sees the patches half the target to the right
  gl_Position.x += 0.5 * float(gl_ViewIndex);
#endif
  gl_PointSize = 1.0;
#ifdef ID
  color = vec4(float(gl_PrimitiveID & 7) * 0.125, float((gl_PrimitiveID >> 3) & 7) * 0.125,
               tc.x, 0.5);
#else
  color = vec4(tc.x, tc.y, TRI ? tc.z : 0.5, 1.0);
#endif
#ifdef WIDE
  for (int k = 0; k < 7; k++)
    wide[k] = vec4(tc, float(k));
#endif
}
