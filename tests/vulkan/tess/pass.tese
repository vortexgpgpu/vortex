#version 450
// The values the control stage made: each corner's own and its neighbour's,
// blended across the patch, and the patch's sum.
layout(quads, equal_spacing, ccw) in;
layout(location = 0) in vec2 t_pos[];
layout(location = 1) in float t_val[];
layout(location = 2) in float t_next[];
layout(location = 3) patch in float p_sum;
layout(location = 0) out vec4 color;
void main() {
  const vec2 o = t_pos[0];
  const vec2 dx = t_pos[1] - o, dy = t_pos[3] - o;
  const vec3 tc = gl_TessCoord;
  gl_Position = vec4(o + tc.x * dx + tc.y * dy, 0.0, 1.0);
  color = vec4(fract(t_val[1] - t_val[0]), fract(t_next[2] + t_next[3]), fract(p_sum), 1.0);
}
