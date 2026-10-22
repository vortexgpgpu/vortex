#version 450

/* Each pixel's own coordinates in red and green, so a texel stored at the
 * wrong address carries the wrong value; blue names the draw. */
layout(push_constant) uniform Push { vec4 rect; };
layout(location = 0) out vec4 o;

void main()
{
   o = vec4(floor(gl_FragCoord.x) / 255.0, floor(gl_FragCoord.y) / 255.0, rect.w, 1.0);
}
