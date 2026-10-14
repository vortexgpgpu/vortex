/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Marks the stencil with its own reference: 1 on odd columns, 2 on even ones,
 * so the EQUAL-to-1 pass that follows admits every other column of the mark. */
#version 450
#extension GL_ARB_shader_stencil_export : require

layout(location = 0) out vec4 out_color;

void main()
{
   gl_FragStencilRefARB = (int(gl_FragCoord.x) & 1) != 0 ? 1 : 2;
   out_color = vec4(0.0, 0.0, 1.0, 1.0);
}
