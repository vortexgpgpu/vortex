#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src_t_ptr =reinterpret_cast<uint8_t*>(arg->src_t_addr);
	auto dst_t_ptr = reinterpret_cast<uint8_t*>(arg->dst_t_addr);
	auto src_u_ptr = reinterpret_cast<uint8_t*>(arg->src_u_addr);
	auto dst_u_ptr = reinterpret_cast<uint8_t*>(arg->dst_u_addr);
	auto src_v_ptr = reinterpret_cast<uint8_t*>(arg->src_v_addr);
	auto dst_v_ptr = reinterpret_cast<uint8_t*>(arg->dst_v_addr);
	auto src_m_ptr = reinterpret_cast<uint8_t*>(arg->src_m_addr);
	auto dst_m_ptr = reinterpret_cast<uint8_t*>(arg->dst_m_addr);
	
	// ===== LOAD ALL TILES FIRST =====
	// This prevents later loads from overwriting T-regs before earlier data is stored
	
	// Test 1: TILE_LOAD_T - Load all 8 T-regs individually
	vx_lt(0, (size_t)(src_t_ptr + 0 * T_TILE_SIZE), 0);
	vx_lt(1, (size_t)(src_t_ptr + 1 * T_TILE_SIZE), 0);
	vx_lt(2, (size_t)(src_t_ptr + 2 * T_TILE_SIZE), 0);
	vx_lt(3, (size_t)(src_t_ptr + 3 * T_TILE_SIZE), 0);
	vx_lt(4, (size_t)(src_t_ptr + 4 * T_TILE_SIZE), 0);
	vx_lt(5, (size_t)(src_t_ptr + 5 * T_TILE_SIZE), 0);
	vx_lt(6, (size_t)(src_t_ptr + 6 * T_TILE_SIZE), 0);
	vx_lt(7, (size_t)(src_t_ptr + 7 * T_TILE_SIZE), 0);
	
	// Store T-tiles immediately while data is still in registers
	vx_st((size_t)(dst_t_ptr + 0 * T_TILE_SIZE), 0, 0);
	vx_st((size_t)(dst_t_ptr + 1 * T_TILE_SIZE), 0, 1);
	vx_st((size_t)(dst_t_ptr + 2 * T_TILE_SIZE), 0, 2);
	vx_st((size_t)(dst_t_ptr + 3 * T_TILE_SIZE), 0, 3);
	vx_st((size_t)(dst_t_ptr + 4 * T_TILE_SIZE), 0, 4);
	vx_st((size_t)(dst_t_ptr + 5 * T_TILE_SIZE), 0, 5);
	vx_st((size_t)(dst_t_ptr + 6 * T_TILE_SIZE), 0, 6);
	vx_st((size_t)(dst_t_ptr + 7 * T_TILE_SIZE), 0, 7);
	
	// Test 2: TILE_LOAD_U - Load all 4 U-regs (covers all 8 T-regs)
	// U-reg 0 maps to T-regs [0, 1]
	vx_lu(0, (size_t)(src_u_ptr + 0 * U_TILE_SIZE), 0);
	vx_st((size_t)(dst_u_ptr + 0 * U_TILE_SIZE), 0, 0);
	vx_st((size_t)(dst_u_ptr + 0 * U_TILE_SIZE + T_TILE_SIZE), 0, 1);
	
	// U-reg 1 maps to T-regs [2, 3]
	vx_lu(1, (size_t)(src_u_ptr + 1 * U_TILE_SIZE), 0);
	vx_st((size_t)(dst_u_ptr + 1 * U_TILE_SIZE), 0, 2);
	vx_st((size_t)(dst_u_ptr + 1 * U_TILE_SIZE + T_TILE_SIZE), 0, 3);
	
	// U-reg 2 maps to T-regs [4, 5]
	vx_lu(2, (size_t)(src_u_ptr + 2 * U_TILE_SIZE), 0);
	vx_st((size_t)(dst_u_ptr + 2 * U_TILE_SIZE), 0, 4);
	vx_st((size_t)(dst_u_ptr + 2 * U_TILE_SIZE + T_TILE_SIZE), 0, 5);
	
	// U-reg 3 maps to T-regs [6, 7]
	vx_lu(3, (size_t)(src_u_ptr + 3 * U_TILE_SIZE), 0);
	vx_st((size_t)(dst_u_ptr + 3 * U_TILE_SIZE), 0, 6);
	vx_st((size_t)(dst_u_ptr + 3 * U_TILE_SIZE + T_TILE_SIZE), 0, 7);
	
	// Test 3: TILE_LOAD_V - Load all 2 V-regs (covers all 8 T-regs)
	// V-reg 0 maps to T-regs [0, 1, 2, 3]
	vx_lv(0, (size_t)(src_v_ptr + 0 * V_TILE_SIZE), 0);
	vx_st((size_t)(dst_v_ptr + 0 * V_TILE_SIZE), 0, 0);
	vx_st((size_t)(dst_v_ptr + 0 * V_TILE_SIZE + 1 * T_TILE_SIZE), 0, 1);
	vx_st((size_t)(dst_v_ptr + 0 * V_TILE_SIZE + 2 * T_TILE_SIZE), 0, 2);
	vx_st((size_t)(dst_v_ptr + 0 * V_TILE_SIZE + 3 * T_TILE_SIZE), 0, 3);
	
	// V-reg 1 maps to T-regs [4, 5, 6, 7]
	vx_lv(1, (size_t)(src_v_ptr + 1 * V_TILE_SIZE), 0);
	vx_st((size_t)(dst_v_ptr + 1 * V_TILE_SIZE), 0, 4);
	vx_st((size_t)(dst_v_ptr + 1 * V_TILE_SIZE + 1 * T_TILE_SIZE), 0, 5);
	vx_st((size_t)(dst_v_ptr + 1 * V_TILE_SIZE + 2 * T_TILE_SIZE), 0, 6);
	vx_st((size_t)(dst_v_ptr + 1 * V_TILE_SIZE + 3 * T_TILE_SIZE), 0, 7);
	
	// Test 4: TILE_LOAD_M - Load all 8 M-regs
	// M-registers store metadata (sparsity patterns/masks)
	vx_lm(0, (size_t)(src_m_ptr + 0 * M_TILE_SIZE), 0);
	vx_lm(1, (size_t)(src_m_ptr + 1 * M_TILE_SIZE), 0);
	vx_lm(2, (size_t)(src_m_ptr + 2 * M_TILE_SIZE), 0);
	vx_lm(3, (size_t)(src_m_ptr + 3 * M_TILE_SIZE), 0);
	vx_lm(4, (size_t)(src_m_ptr + 4 * M_TILE_SIZE), 0);
	vx_lm(5, (size_t)(src_m_ptr + 5 * M_TILE_SIZE), 0);
	vx_lm(6, (size_t)(src_m_ptr + 6 * M_TILE_SIZE), 0);
	vx_lm(7, (size_t)(src_m_ptr + 7 * M_TILE_SIZE), 0);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, nullptr, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
