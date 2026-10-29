// Shared by the mesh test's task and mesh shaders: the order every case's
// primitives are numbered in, and the screen cell each one is drawn in. The
// host (main.c) builds its expectation from the same functions.

// The target is 64x64 pixels in 4x4 cells of 16x16.
const uint CELLS = 16u;

// Primitives mesh workgroup w emits when its count is not static: 0 to 32.
uint wg_prims(uint w)
{
   return (w * 7u + 3u) % 33u;
}

// The cell primitive k of workgroup w is drawn in.
uint prim_cell(uint w, uint k)
{
   return (w * 13u + k * 5u) % CELLS;
}

// The colour primitive k of workgroup w is drawn in: never the clear colour.
uint prim_colour(uint w, uint k)
{
   return 0xff000000u | ((w << 8) | k) + 1u;
}

vec4 ndc(vec2 px)
{
   return vec4(px / 32.0 - 1.0, 0.5, 1.0);
}

// A triangle in cell c whose slanted edge runs two pixels across per pixel
// down, so no pixel centre lies on an edge.
vec4 cell_vertex(uint c, uint v)
{
   const vec2 o = vec2(float(c % 4u) * 16.0, float(c / 4u) * 16.0);
   const vec2 p[3] = vec2[3](vec2(2.0, 2.0), vec2(14.0, 2.0), vec2(2.0, 8.0));
   return ndc(o + p[v]);
}

// The pixel big.mesh draws point k of workgroup w on.
uint big_pixel(uint w, uint k)
{
   return ((w * 32u + k) * 37u) % 4096u;
}

// gl_WorkGroupID as one index, x varying fastest.
uint wg_linear()
{
   return gl_WorkGroupID.x +
          gl_NumWorkGroups.x * (gl_WorkGroupID.y + gl_NumWorkGroups.y * gl_WorkGroupID.z);
}
