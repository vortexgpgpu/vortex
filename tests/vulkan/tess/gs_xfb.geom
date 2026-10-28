#version 450
// The tessellated triangles passed on unchanged, their colour captured.
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 color[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) out vec4 o_color;
void main() {
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;
    o_color = color[i];
    EmitVertex();
  }
}
