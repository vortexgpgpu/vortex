#version 450
// Quads at level 64 everywhere, each domain point marking its own slot of
// `seen` -- (gl_PrimitiveID * 65 + j) * 65 + i for the point at (i, j) / 64
// -- with the slot plus one, through records as wide as an evaluation stage
// may make them: 32 vec4 outputs.
layout(quads, equal_spacing, ccw) in;
layout(location = 0) in vec2 t_pos[];
layout(location = 0) out vec4 wide[32];
layout(set = 0, binding = 0, std430) buffer Seen { uint seen[]; };
void main() {
  const vec3 tc = gl_TessCoord;
  const uint i = uint(round(tc.x * 64.0)), j = uint(round(tc.y * 64.0));
  const uint slot = (uint(gl_PrimitiveID) * 65u + j) * 65u + i;
  seen[slot] = slot + 1u;
  gl_Position = vec4(t_pos[0], 0.0, 1.0);
  for (int k = 0; k < 32; k++)
    wide[k] = vec4(tc, float(k));
}
