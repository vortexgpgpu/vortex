/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * The same triangle as imagedraw.vert, with one storage-image write per vertex.
 *
 * A draw relocates the vertex stage's descriptors through the same loop as the
 * fragment stage's, so an image bound to a vertex shader needs the same
 * treatment and fails the same way when it does not get it. Nothing else in
 * this suite writes an image from a vertex shader.
 *
 * Each vertex stores a different red so the check cannot pass on an image
 * filled with one value, and every channel avoids 0 so it cannot pass on an
 * image that was merely left as allocated. */
#version 450

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D dst;

layout(location = 0) out vec2 v_uv;

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  3.0 : -1.0,
                 (gl_VertexIndex == 2) ?  3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
   v_uv = (p + 1.0) * 0.5;

   /* Exact 1/255 steps, so the stored bytes are the same under the device's
    * truncation and llvmpipe's rounding: reds 64, 128, 192. */
   imageStore(dst, ivec2(gl_VertexIndex, 0),
              vec4(float((gl_VertexIndex + 1) * 64), 128.0, 64.0, 255.0) / 255.0);
}
