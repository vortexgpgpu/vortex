/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A layer coordinate past the last layer of a three-layer array. Vulkan
 * clamps it to the last layer, so this must read layer 2 -- and must not
 * read past the resource. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DArray tex;

void main()
{
   vec4 t = texture(tex, vec3(v_uv, 7.0));
   vec4 want = vec4(120.0 / 255.0, 50.0 / 255.0, 80.0 / 255.0, 1.0);  /* layer 2 */
   bool ok = all(lessThan(abs(t - want), vec4(0.02)));
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
