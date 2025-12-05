#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// T-reg: 1KB (16x16 fp32 elements)
#define T_TILE_SIZE 1024

// U-reg: 2KB (2 x T-reg)
#define U_TILE_SIZE 2048

// V-reg: 4KB (4 x T-reg)
#define V_TILE_SIZE 4096

// M-reg: 128B (16x16 4-bit elements, 2 per byte)
#define M_TILE_SIZE 128

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
