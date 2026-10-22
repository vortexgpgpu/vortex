#version 450

/* The texture at this pixel's texel centre, nearest-filtered. */
layout(binding = 0) uniform sampler2D tex;
layout(push_constant) uniform Push { vec2 inv_size; };
layout(location = 0) out vec4 o;

void main()
{
   o = texture(tex, gl_FragCoord.xy * inv_size);
}
