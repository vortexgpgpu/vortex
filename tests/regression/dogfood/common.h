#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t testid;
  uint32_t num_tasks;  
  uint32_t task_size;  
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

// imul64: every 64-bit multiply form (MUL, MULH, MULHSU, MULHU, MULW) in a
// dependent chain, folded to 32 bits; the kernel and the host check share it.
#define IMUL64_ROUNDS 8

#ifdef __SIZEOF_INT128__
__extension__ typedef __int128 imul64_s128;
__extension__ typedef unsigned __int128 imul64_u128;
#endif

static inline uint64_t imul64_mulhu(uint64_t a, uint64_t b) {
#ifdef __SIZEOF_INT128__
  return (uint64_t)(((imul64_u128)a * b) >> 64);
#else
  uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
  uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
  return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
}

static inline uint64_t imul64_mulh(uint64_t a, uint64_t b) {
#ifdef __SIZEOF_INT128__
  return (uint64_t)(((imul64_s128)(int64_t)a * (imul64_s128)(int64_t)b) >> 64);
#else
  uint64_t h = imul64_mulhu(a, b);
  if ((int64_t)a < 0) h -= b;
  if ((int64_t)b < 0) h -= a;
  return h;
#endif
}

static inline uint64_t imul64_mulhsu(uint64_t a, uint64_t b) {
#ifdef __SIZEOF_INT128__
  return (uint64_t)(((imul64_s128)(int64_t)a * (imul64_s128)b) >> 64);
#else
  uint64_t h = imul64_mulhu(a, b);
  if ((int64_t)a < 0) h -= b;
  return h;
#endif
}

static inline uint32_t imul64_mix(uint32_t s0, uint32_t s1) {
  uint64_t x = ((uint64_t)s0 << 32) | s1;
  uint64_t y = ((uint64_t)s1 << 32) | (uint32_t)~s0;
  uint64_t h = 0;
  for (int r = 0; r < IMUL64_ROUNDS; ++r) {
    uint64_t lo  = x * y;
    uint64_t hs  = imul64_mulh(x, y);
    uint64_t hu  = imul64_mulhu(x, y);
    uint64_t hsu = imul64_mulhsu(x, y);
    int64_t  w   = (int32_t)((uint32_t)x * (uint32_t)y);
    h ^= lo ^ (hs << 1) ^ (hu << 2) ^ (hsu << 3) ^ (uint64_t)w;
    x = lo ^ hsu ^ (uint64_t)r;
    y = hs + hu + (uint64_t)w;
  }
  return (uint32_t)(h ^ (h >> 32));
}

#endif