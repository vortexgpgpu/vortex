#version 460
layout(location = 0) flat in uint vcol;
layout(location = 0) out vec4 o;

void main()
{
   o = unpackUnorm4x8(vcol);
}
