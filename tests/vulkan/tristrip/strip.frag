/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Hello-triangle fragment shader. In Phase 3 the fragment stage
 * stays on llvmpipe's CPU path; only the vertex stage moves to
 * Vortex. */
#version 450

layout(location = 0) in  vec3 v_color;
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(v_color, 1.0);
}
