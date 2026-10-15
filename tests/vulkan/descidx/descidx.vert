#version 450

/* One triangle over the whole target, each vertex recording itself through a
 * fixed storage-buffer descriptor. */

layout(std430, binding = 5) buffer VsOut { uint v[]; } vsout;

void main()
{
   const vec2 pos[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
   vsout.v[gl_VertexIndex] = 0x1000u + uint(gl_VertexIndex);
   gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
}
