#version 450

/* Storage image and buffer accesses from the fragment stage of a 1x1 pass:
 * the one invocation elects itself with a compare-and-swap, loads a storage
 * image texel, reports it, then adds to the elected word -- the CTS's
 * descriptor-check pattern. The pass has no attachments, so the stores are the
 * fragment stage's only output. */

layout(binding = 0, r32ui) uniform uimage2D img;
layout(std430, binding = 1) buffer Out { uint flag; uint seen; } outb;

void main()
{
   const uint first = atomicCompSwap(outb.flag, 0u, 1u);
   if (first == 0u) {
      outb.seen = imageLoad(img, ivec2(0, 0)).x;
      atomicAdd(outb.flag, 1u);
   }
}
