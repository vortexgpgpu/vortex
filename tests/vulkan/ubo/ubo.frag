/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * UBO/push-constant descriptor test — fragment stage (W7). The colour
 * is read entirely from a uniform buffer (set 0, binding 0) scaled by
 * a push constant, so a correct render proves the fragment-shader
 * descriptor path (UBO read + push-constant read) works on device —
 * independent of vertex interpolation. */
#version 450

layout(set = 0, binding = 0) uniform UBO {
   vec4 color;
} ubo;

layout(push_constant) uniform PC {
   float scale;
} pc;

layout(location = 0) in  vec3 v_color;   /* unused: isolates the UBO path */
layout(location = 0) out vec4 out_color;

void main()
{
   out_color = vec4(ubo.color.rgb * pc.scale, 1.0);
}
