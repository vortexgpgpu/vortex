/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Quad q (vertices 6q..6q+5) goes into framebuffer layer q: quad 0 covers NDC
 * [-1, 0]^2 and quad 1 [0, 1]^2, so a layer that receives the other's quad
 * shows it in the wrong corner. */
#version 450
#extension GL_ARB_shader_viewport_layer_array : require

const vec2 corners[6] = vec2[](
   vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
   vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
);

void main()
{
   const int q = gl_VertexIndex / 6;
   gl_Position = vec4(corners[gl_VertexIndex % 6] + vec2(float(q) - 1.0), 0.0, 1.0);
   gl_Layer = q;
}
