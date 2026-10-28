#version 450
// Constant outer levels and no inner ones: the inner levels a quad reads are
// 0, so the levels are not the host's to process.
layout(vertices = 4) out;
layout(location = 0) in vec2 v_pos[];
layout(location = 0) out vec2 t_pos[];
void main() {
  t_pos[gl_InvocationID] = v_pos[gl_InvocationID];
  if (gl_InvocationID == 0) {
    gl_TessLevelOuter[0] = 3.0;
    gl_TessLevelOuter[1] = 2.0;
    gl_TessLevelOuter[2] = 4.0;
    gl_TessLevelOuter[3] = 5.0;
  }
}
