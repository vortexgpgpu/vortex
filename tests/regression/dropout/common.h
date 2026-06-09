#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_points;
  float dropout_p;
  float multiplier;
  uint64_t src0_addr;
  uint64_t dst_addr; 
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
