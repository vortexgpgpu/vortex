/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The same custom border on an R8 texture. The border replaces an R8 texel, so
 * it has an R8 texel's components: red from the border, green and blue 0 and
 * alpha 1, whatever the border colour holds there. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   const vec4 border = vec4(0.2, 0.0, 0.0, 1.0);
   const vec4 texel  = vec4(40.0 / 255.0, 0.0, 0.0, 1.0);
   const float tol   = 1.5 / 255.0;

   vec4 inside = texture(tex, vec2(0.5, 0.5));
   vec4 left   = texture(tex, vec2(-0.5, 0.5));
   vec4 blue   = textureGather(tex, vec2(-0.5, 0.5), 2);
   vec4 alpha  = textureGather(tex, vec2(-0.5, 0.5), 3);

   bool ok = all(lessThan(abs(inside - texel), vec4(tol)))
          && all(lessThan(abs(left - border), vec4(tol)))
          && all(lessThan(abs(blue), vec4(tol)))
          && all(lessThan(abs(alpha - vec4(1.0)), vec4(tol)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
