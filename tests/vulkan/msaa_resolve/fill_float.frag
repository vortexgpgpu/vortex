/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A distinct R32F value in every sample, x + 32y + k/4: every value and the
 * average of the four (x + 32y + 3/8) is exact in float, so a resolve must
 * reproduce it bit for bit. Keep in step with expect_float() in main.c. */
#version 450

layout(location = 0) out vec4 out_color;

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   out_color = vec4(float(p.x + 32 * p.y) + 0.25 * float(gl_SampleID),
                    0.0, 0.0, 1.0);
}
