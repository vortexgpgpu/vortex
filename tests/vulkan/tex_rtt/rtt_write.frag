#version 450
#extension GL_GOOGLE_include_directive : require

/* Round k's pattern, rendered. */
layout(push_constant) uniform Push { uint k; };
layout(location = 0) out vec4 o;

#include "rtt_pattern.glsl"

void main()
{
   o = unpackUnorm4x8(pattern(uint(gl_FragCoord.x), uint(gl_FragCoord.y), k));
}
