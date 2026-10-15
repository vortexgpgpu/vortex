#version 450

/* One triangle over the whole target. */
void main()
{
   const vec2 pos[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
   gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
}
