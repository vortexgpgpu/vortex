/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Triangle t covers the centre of pixel t % 65536 of a 256x256 target and no
// other, in a colour of its own: the draw's later triangles overwrite earlier
// ones, so the image shows which came last. Drawn as points, it hands the
// geometry stage nothing but its primitive id.
layout(push_constant) uniform pc_t { uint points; } pc;
layout(location = 0) flat out vec4 vcol;
void main()
{
   uint t = pc.points != 0u ? uint(gl_VertexIndex) : uint(gl_VertexIndex) / 3u;
   uint k = uint(gl_VertexIndex) % 3u;
   uint p = t % 65536u;
   vec2 c = vec2(float(p % 256u), float(p / 256u)) + 0.1;
   c += k == 1u ? vec2(1.0, 0.0) : k == 2u ? vec2(0.0, 1.0) : vec2(0.0);
   gl_Position = vec4(c / 128.0 - 1.0, 0.0, 1.0);
   gl_PointSize = 1.0;
   vcol = vec4(float(t & 255u), float((t >> 8) & 255u), float(t >> 16), 255.0) / 255.0;
}
