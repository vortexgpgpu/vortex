#version 450
// The tessellated triangles passed on by the first of 32 invocations, each
// invocation declaring room for far more output than any writes.
layout(triangles, invocations = 32) in;
layout(triangle_strip, max_vertices = 128) out;
layout(location = 0) in vec4 color[];
layout(location = 0) out vec4 o_color;
void main() {
  if (gl_InvocationID != 0)
    return;
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;
    o_color = color[i];
    EmitVertex();
  }
}
