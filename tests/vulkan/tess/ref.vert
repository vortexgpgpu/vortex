#version 450
// The reference: the primitives the host tessellated, drawn as a list.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 col;
layout(location = 0) out vec4 color;
void main() {
  gl_Position = vec4(pos, 0.0, 1.0);
  gl_PointSize = 1.0;
  color = col;
}
