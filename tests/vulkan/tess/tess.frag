#version 450
layout(location = 0) in vec4 color;
layout(location = 0) out vec4 o;
void main() {
  o = gl_FrontFacing ? color : color * 0.5;
}
