#version 450
// Control points pass through; the instance moves its patches down a row.
// With LIVE set, every other instance's patches get zero levels, which discard
// them.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 lo;
layout(location = 2) in vec2 li;
layout(location = 0) out vec2 v_pos;
layout(location = 1) out vec4 v_lo;
layout(location = 2) out vec2 v_li;
layout(push_constant) uniform pc_t { vec2 inst_step; } pc;
layout(constant_id = 7) const int LIVE = -1;
void main() {
  v_pos = pos + float(gl_InstanceIndex) * pc.inst_step;
  v_lo = LIVE < 0 || gl_InstanceIndex == LIVE ? lo : vec4(0.0);
  v_li = li;
}
