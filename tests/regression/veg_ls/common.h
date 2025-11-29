#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// T-reg: 2KB (16x32 4-byte elements)
#define T_TILE_SIZE 2048

// U-reg: 4KB (2 x T-reg)
#define U_TILE_SIZE 4096

// V-reg: 8KB (4 x T-reg)
#define V_TILE_SIZE 8192

// M-reg: 256B (16x32 4-bit elements)
#define M_TILE_SIZE 256

// Number of tiles to test for each register type
#define NUM_T_TILES 8  // Test all 8 T-regs
#define NUM_U_TILES 4  // Test all 4 U-regs (covers all 8 T-regs)
#define NUM_V_TILES 2  // Test all 2 V-regs (covers all 8 T-regs)
#define NUM_M_TILES 8  // Test all 8 M-regs

typedef struct {
  uint64_t src_t_addr;  // Source address for T tiles
  uint64_t dst_t_addr;  // Destination address for T tiles
  uint64_t src_u_addr;  // Source address for U tiles
  uint64_t dst_u_addr;  // Destination address for U tiles
  uint64_t src_v_addr;  // Source address for V tiles
  uint64_t dst_v_addr;  // Destination address for V tiles
  uint64_t src_m_addr;  // Source address for M tiles
  uint64_t dst_m_addr;  // Destination address for M tiles
} kernel_arg_t;

#endif
