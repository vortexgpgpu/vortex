/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Forwards the vertex stage's subgroup report to the colour target, byte-encoded
 * so an 8-bit UNORM attachment carries each component exactly.
 *
 * The quarter-step bias is what makes that true on both rasterizers. v/255.0 is
 * exact only if the conversion back rounds; the device truncates instead -- a
 * deliberate choice, for bit-exactness with the fixed-function output merger --
 * so every integer would come back one short. Biasing into the middle of the
 * step lands on v whether the conversion truncates or rounds to nearest.
 *
 * This was invisible while a flat input forced the fragment stage onto the host
 * rasterizer, which rounds. It is an encoding fix, not a relaxed expectation:
 * the values checked are unchanged.
 */
#version 450

layout(location = 0) flat in uvec4 v_sg;
layout(location = 0) out vec4 o_color;

void main()
{
   o_color = (vec4(v_sg) + 0.25) / 255.0;
}
