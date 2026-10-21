/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Quad q (vertices 6q..6q+5) goes through viewport q, spanning NDC
 * [-1, hi] on both axes. */
#version 450
#extension GL_ARB_shader_viewport_layer_array : require

layout(push_constant) uniform PC { float hi; } pc;

const vec2 corners[6] = vec2[](
   vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
   vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
);

void main()
{
   const vec2 c = corners[gl_VertexIndex % 6];
   gl_Position = vec4(mix(vec2(-1.0), vec2(pc.hi), c), 0.0, 1.0);
   gl_ViewportIndex = gl_VertexIndex / 6;
}
