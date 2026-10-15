/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * samplerCubeShadow on a D32F cube whose face f holds depth (f + 1) / 8,
 * compare LESS: along each face's axis a reference just below that face's
 * depth passes and one just above fails, so a wrong face selection breaks the
 * pattern. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform samplerCubeShadow tex;

void main()
{
   const vec3 dirs[6] = vec3[](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                               vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
   bool ok = true;
   for (int f = 0; f < 6; f++) {
      float d = float(f + 1) / 8.0;
      ok = ok && texture(tex, vec4(dirs[f], d - 0.0625)) > 0.5 &&
                 texture(tex, vec4(dirs[f], d + 0.0625)) < 0.5;
   }
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
