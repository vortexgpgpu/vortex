#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require

/* The compute shader's per-lane descriptor indexing, from the fragment stage:
 * pixel (x, y) of a 16x16 attachment-less pass is lane i = y * 16 + x. */

layout(std430, binding = 0) readonly buffer Src { uint v[]; } src[4];
layout(std430, binding = 1) buffer Dst { uint v[]; } dst[4];
layout(std430, binding = 2) buffer Out { uint v[]; } outb;
layout(r32ui, binding = 3) uniform uimage2D img[4];
layout(binding = 4) uniform utextureBuffer tbo[4];
layout(binding = 6) uniform readonly uimage2D rimg[4];
layout(binding = 7) uniform utexture2D simg[4];
layout(binding = 8) uniform sampler smp[2];

void main()
{
   uint i = uint(gl_FragCoord.y) * 16u + uint(gl_FragCoord.x);
   uint k = i & 3u;
   ivec2 c = ivec2(int(i >> 2) & 7, int(i >> 5));
   outb.v[i] = src[nonuniformEXT(k)].v[i >> 2] + imageLoad(img[nonuniformEXT(k)], c).x +
               texelFetch(tbo[nonuniformEXT(k)], int(i >> 2)).x +
               imageLoad(rimg[nonuniformEXT(k)], c).x;
   /* One period past the right edge: smp[0] clamps to the last column,
    * smp[1] repeats back to column c.x. */
   vec2 uv = vec2((float(c.x) + 0.5) / 8.0 + 1.0, (float(c.y) + 0.5) / 8.0);
   outb.v[i] += textureLod(usampler2D(simg[nonuniformEXT(k)], smp[1]), uv, 0.0).x +
                textureLod(usampler2D(simg[2], smp[nonuniformEXT(k & 1u)]), uv, 0.0).x;
   dst[nonuniformEXT(k)].v[i >> 2] = i;
   imageStore(img[nonuniformEXT(k)], c, uvec4(i));
}
