/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Samples all six cube faces from one fragment, each through a direction
 * whose major axis picks that face. Face f is a solid (40f+30, 25f+5,
 * 200-30f), so a wrong face-select or a wrong layer order shows up as a
 * colour mismatch rather than as noise. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform samplerCube tex;

vec4 face_color(int f)
{
   return vec4(float(40 * f + 30) / 255.0,
               float(25 * f +  5) / 255.0,
               float(200 - 30 * f) / 255.0, 1.0);
}

void main()
{
   /* Vulkan cube layer order: +X, -X, +Y, -Y, +Z, -Z. */
   vec3 dirs[6] = vec3[](vec3( 1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0),
                         vec3( 0.0,  1.0,  0.0), vec3( 0.0, -1.0,  0.0),
                         vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0, -1.0));
   bool ok = true;
   for (int f = 0; f < 6; f++) {
      vec4 t = texture(tex, dirs[f]);
      ok = ok && all(lessThan(abs(t - face_color(f)), vec4(0.02)));
   }
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
