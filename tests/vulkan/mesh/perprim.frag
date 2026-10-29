#version 460
#extension GL_EXT_mesh_shader : require

// The primitive's colour with its gl_PrimitiveID folded into blue and the
// provoking vertex's value into alpha, so a lost per-primitive value, a wrong
// primitive id or the wrong vertex each changes the pixel.
layout(location = 0) flat in uint vcol;
layout(location = 1) perprimitiveEXT flat in uint pcol;
layout(location = 0) out vec4 o;

void main()
{
   const uint c = (pcol & 0x0000ffffu) | ((uint(gl_PrimitiveID) & 0xffu) << 16) |
                  ((vcol & 0xffu) << 24);
   o = unpackUnorm4x8(c);
}
