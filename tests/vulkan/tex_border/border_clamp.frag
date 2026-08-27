/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * CLAMP_TO_BORDER: a coordinate outside [0,1] returns the sampler's border
 * colour. The texture is a solid colour distinct from the border, so a wrap
 * mode that fell back to clamp-to-edge returns the texel instead and fails. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   const vec4 border = vec4(1.0, 1.0, 1.0, 1.0);           /* opaque white */
   const vec4 texel  = vec4(0.0, 40.0 / 255.0, 1.0, 1.0);  /* the texture  */

   vec4 inside = texture(tex, vec2(0.5, 0.5));
   vec4 left   = texture(tex, vec2(-0.5, 0.5));
   vec4 right  = texture(tex, vec2( 1.5, 0.5));
   vec4 above  = texture(tex, vec2( 0.5, -0.5));

   bool ok = all(lessThan(abs(inside - texel), vec4(0.02)))
          && all(lessThan(abs(left  - border), vec4(0.02)))
          && all(lessThan(abs(right - border), vec4(0.02)))
          && all(lessThan(abs(above - border), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
