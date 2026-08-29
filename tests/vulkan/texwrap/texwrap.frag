/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Reports what three probe coordinates sampled, rather than judging them: the
 * host holds the expected triple for each address mode, so all four modes are
 * compared against one another in one place and two modes returning the same
 * answer is visible as such.
 *
 * The texture is 4 texels wide with a distinct value per column, so a probe
 * outside [0,1) names the texel the wrap arithmetic chose. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   /* Each probe sits on a texel centre so NEAREST has no boundary to round,
    * and is jittered by well under half a texel (0.125) so lanes carry
    * different addresses while still selecting the same texel. */
   float j = (v_uv.x - 0.5) * 0.1;

   float below  = texture(tex, vec2(-0.375 + j, 0.5)).r;   /* texel index -2 */
   float above  = texture(tex, vec2( 1.375 + j, 0.5)).r;   /* texel index  5 */
   float inside = texture(tex, vec2( 0.625 + j, 0.5)).r;   /* texel index  2 */

   out_color = vec4(below, above, inside, 1.0);
}
