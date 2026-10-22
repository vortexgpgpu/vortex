#version 450

/* A rectangle spanning rect.x..rect.y in x and the whole height, at depth
 * rect.z: two triangles from the vertex index alone. */
layout(push_constant) uniform Push { vec4 rect; };

void main()
{
   const vec2 c[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1),
                            vec2(1, 0), vec2(1, 1), vec2(0, 1));
   vec2 t = c[gl_VertexIndex];
   gl_Position = vec4(mix(rect.x, rect.y, t.x), mix(-1.0, 1.0, t.y), rect.z, 1.0);
}
