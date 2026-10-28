#version 450
/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A colour exact in RGBA8 that names the pixel and the pass.
 */
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Params { uint pass; float depth; } pc;

void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   out_color = vec4(float(p.x & 255u), float(p.y & 255u),
                    float((p.x >> 8) + (p.y >> 8) * 8u + pc.pass * 64u), 255.0) / 255.0;
}
