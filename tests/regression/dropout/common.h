#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif


//==============================================================================
// DFV (Design-for-Verification) CSR Definitions
//==============================================================================
#define VX_CSR_DFV_CTRL           0x7C0
#define VX_CSR_DFV_ICACHE_STALL   0x7C1
#define VX_CSR_DFV_RANDOM_SEED    0x7C2
#define VX_CSR_DFV_SET_THRESHOLD   0x7C3
#define VX_CSR_DFV_DCACHE_STALL   0x7C4
#define VX_CSR_DFV_WRITEBACK_STALL 0x7C5 // Writeback stall enable (bit 0)
#define VX_CSR_DFV_FILL_STALL      0x7C6 // Cache fill stall enable (bit 0)
#define VX_CSR_DFV_RELEASE_THRESHOLD 0x7C7 // Release probability (0-255)
#define VX_CSR_DFV_RELEASE_SEED    0x7C8 // LFSR2 seed for release timing
#define VX_CSR_DFV_RELEASE_DELAY   0x7C9 // Per-point release delay [3:0]=ic [7:4]=dc [11:8]=wb [15:12]=fill
#define VX_CSR_DFV_RELEASE_FOREVER 0x7CA // When 1: once released, stalls stay off permanently
#define VX_CSR_DFV_THROTTLE_THRESHOLD 0x7CB // Throttle counter threshold (16-bit)

typedef struct {
  uint32_t num_points;
  float dropout_p;
  float multiplier;
  uint64_t src0_addr;
  uint64_t dst_addr; 
  uint32_t enable_dfv_test;
} kernel_arg_t;

unsigned int WangHash(unsigned int s){
	s = (s^61) ^ (s >> 16);
	s *= 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d;
	s = s ^ (s >> 15);
  return s;
}
unsigned int RandomInt(unsigned int s){
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

float RandomFloat(unsigned int s){
	return RandomInt(s) * 2.3283064365387e-10f;
}
#endif
