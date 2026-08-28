/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A storage image written from the fragment stage.
 *
 * The compute path relocates all three descriptor kinds a shader can reach --
 * buffer, acceleration structure and image. A draw relocates only buffers, so
 * the image descriptor a fragment shader is handed still holds the host pointer
 * the CPU rasterizer put there. Nothing in this suite bound a storage image to
 * a draw, so that gap had nothing to reveal it.
 *
 * The written colour has a distinct value per channel, none of them 0 or 255,
 * so a texel that arrives cleared, saturated or channel-swapped is told apart
 * from one that arrives right. The colour attachment is written too, and
 * checked, so a failure says whether the draw ran at all. */
#version 450

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D dst;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
   ivec2 p = ivec2(v_uv * 64.0);

   /* Written as exact 1/255 steps: 0.5 and 0.25 land on 127.5 and 63.75,
    * which the device truncates and llvmpipe rounds, and a check that
    * tolerated that difference would tolerate a real one the same size. */
   imageStore(dst, p, vec4(255.0, 128.0, 64.0, 255.0) / 255.0);

   o_color = vec4(0.0, 1.0, 0.0, 1.0);
}
