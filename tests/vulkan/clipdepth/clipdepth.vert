/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A full-screen quad whose depth runs from z.x at the left edge to z.y at the
 * right. Mode 1 clips at x = 0 with gl_ClipDistance; mode 2 culls the quad's
 * second triangle with gl_CullDistance. */
#version 450

layout(push_constant) uniform PC {
   vec2 z;
   int  mode;
} pc;

out gl_PerVertex {
   vec4  gl_Position;
   float gl_ClipDistance[1];
   float gl_CullDistance[1];
};

const vec2 quad[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
                            vec2(-1,  1), vec2(1, -1), vec2( 1, 1));

void main()
{
   vec2 p = quad[gl_VertexIndex];
   gl_Position = vec4(p, mix(pc.z.x, pc.z.y, p.x * 0.5 + 0.5), 1.0);
   gl_ClipDistance[0] = (pc.mode == 1) ? p.x : 1.0;
   /* The second triangle has the distance negative at all three vertices and
    * is culled; the first has it negative at one, which keeps it. */
   gl_CullDistance[0] = (pc.mode == 2 && (gl_VertexIndex >= 3 || gl_VertexIndex == 0))
                      ? -1.0 : 1.0;
}
