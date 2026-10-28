#version 450
// The tessellated primitives passed on unchanged.
layout(points) in;
layout(points, max_vertices = 1) out;
layout(location = 0) in vec4 color[];
layout(location = 0) out vec4 o_color;
void main() {
  for (int i = 0; i < 1; i++) {
    gl_Position = gl_in[i].gl_Position;
    gl_PointSize = 1.0;
    o_color = color[i];
    EmitVertex();
  }
}
