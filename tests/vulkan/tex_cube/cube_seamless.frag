/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * LINEAR samples of a cube with six solid faces, on two face edges and a
 * corner. A cube filters seamlessly: an edge direction's footprint straddles two
 * faces equally, and a corner's takes the three faces meeting there. A sampler
 * that clamps within a face returns that one face's colour instead. */
#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform samplerCube tex;

vec4 face_color(int f)
{
   return vec4(float(40 * f + 30) / 255.0,
               float(25 * f +  5) / 255.0,
               float(200 - 30 * f) / 255.0, 1.0);
}

bool near(vec4 a, vec4 b)
{
   return all(lessThan(abs(a - b), vec4(0.02)));
}

void main()
{
   /* Vulkan cube layer order: +X, -X, +Y, -Y, +Z, -Z. */
   bool ok = near(texture(tex, vec3(1.0, 0.0, 1.0)), (face_color(0) + face_color(4)) * 0.5)
          && near(texture(tex, vec3(-1.0, 1.0, 0.0)), (face_color(1) + face_color(2)) * 0.5)
          && near(texture(tex, vec3(1.0, 1.0, 1.0)),
                  (face_color(0) + face_color(2) + face_color(4)) / 3.0);
   out_color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
