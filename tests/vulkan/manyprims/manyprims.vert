/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Triangle t of the draw, built from gl_VertexIndex, lies inside one pixel.
 * Every 32nd triangle covers its pixel's centre, at a pixel of its own; the rest
 * sit in a pixel's lower-right corner, past the centre, and cover no sample.
 * Each carries its own index t to the fragment stage. The host's expectation in
 * main.c mirrors this placement.
 *
 * Vertex 1 lies just in front of the near plane (z < 0), so the near clip cuts
 * a sliver off that corner, clear of the pixel centre, and every triangle sets
 * up as two: the draw's record count is twice its triangle count. */
#version 450

layout(location = 0) flat out uint v_tri;

/* The render target (WIDTH x HEIGHT in main.c) and the covering stride. */
const uint WIDTH  = 64u;
const uint HEIGHT = 64u;
const uint COVER_STRIDE = 32u;
/* Triangles per instance: two instances, a count the pass size does not divide. */
const uint TRIS_PER_INST = 16896u;

void main()
{
   const uint t = uint(gl_InstanceIndex) * TRIS_PER_INST + uint(gl_VertexIndex) / 3u;
   const uint k = uint(gl_VertexIndex) % 3u;
   const bool covers = (t % COVER_STRIDE) == COVER_STRIDE - 1u;
   const uint pixel = covers ? t / COVER_STRIDE : t;
   const float x = float(pixel % WIDTH);
   const float y = float((pixel / WIDTH) % HEIGHT);
   const float lo = covers ? 0.1 : 0.6;
   const float hi = 0.95;
   const vec2 p = vec2(x + (k == 1u ? hi : lo), y + (k == 2u ? hi : lo));
   gl_Position = vec4(2.0 * p.x / float(WIDTH) - 1.0,
                      2.0 * p.y / float(HEIGHT) - 1.0, k == 1u ? -0.05 : 0.5, 1.0);
   v_tri = t;
}
