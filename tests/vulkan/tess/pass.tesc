#version 450
// Every invocation writes its control point and a value, then, after the
// barrier, reads its neighbours' values back: the patch's sum goes out per
// patch, and each point carries the next one's value.
layout(vertices = 4) out;
layout(location = 0) in vec2 v_pos[];
layout(location = 1) in vec4 v_lo[];
layout(location = 2) in vec2 v_li[];
layout(location = 0) out vec2 t_pos[];
layout(location = 1) out float t_val[];
layout(location = 2) out float t_next[];
layout(location = 3) patch out float p_sum;
void main() {
  t_pos[gl_InvocationID] = v_pos[gl_InvocationID];
  t_val[gl_InvocationID] = v_pos[gl_InvocationID].x * 2.0 + float(gl_InvocationID) * 0.125;
  barrier();
  t_next[gl_InvocationID] = t_val[(gl_InvocationID + 1) & 3];
  if (gl_InvocationID == 0) {
    p_sum = t_val[0] + t_val[1] + t_val[2] + t_val[3];
    gl_TessLevelOuter[0] = v_lo[0].x;
    gl_TessLevelOuter[1] = v_lo[0].y;
    gl_TessLevelOuter[2] = v_lo[0].z;
    gl_TessLevelOuter[3] = v_lo[0].w;
    gl_TessLevelInner[0] = v_li[0].x;
    gl_TessLevelInner[1] = v_li[0].y;
  }
}
