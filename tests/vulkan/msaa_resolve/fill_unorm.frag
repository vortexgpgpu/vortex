/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A distinct RGBA8 value in every sample: reading gl_SampleID runs the shader
 * once per sample. The bytes are chosen so the four samples of a channel
 * differ, and their average is an integer in red, green and alpha and falls
 * between two codes in blue, where the resolve's rounding is free to go
 * either way. Keep in step with expect_unorm() in main.c. */
#version 450

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PC {
   int seed;
};

void main()
{
   ivec2 p = ivec2(gl_FragCoord.xy);
   int k = gl_SampleID;
   vec4 bytes = vec4(float(16 * k + p.x + seed),
                     float(50 * k + p.y),
                     k == 0 ? 255.0 : 0.0,
                     float(255 - 60 * k));
   out_color = bytes / 255.0;
}
