/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
#extension GL_EXT_multiview : require
// gl_ViewIndex read by the geometry stage and handed to the fragment stage,
// which does not read it itself.
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) flat out uint view;
void main()
{
   for (int k = 0; k < 3; k++) {
      gl_Position = gl_in[k].gl_Position;
      view = gl_ViewIndex;
      EmitVertex();
   }
}
