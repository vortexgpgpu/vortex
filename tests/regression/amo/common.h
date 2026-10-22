#ifndef _COMMON_H_
#define _COMMON_H_

// Slots in the compare-and-swap ladder. The shared buffer is sized from the
// hart count, and one hart still gives 64 bytes, so 16 words always fit
// whatever the launch shape or the iteration count.
#define CAS_LADDER_SLOTS 16

// Slots in the local-memory compare-and-swap ladder (64 bytes of a group's
// local memory), and the marker a group leader folds into its published win
// count so the host can count groups. Wins never exceed the slot count, so
// the marker is unambiguous.
#define CAS_LMEM_SLOTS      16
#define CAS_LMEM_GROUP_MARK 0x10000u

// Starting value of the compare-and-swap counters: nonzero, so no comparand a
// swap uses is ever zero.
#define CAS_COUNTER_BASE    0x5a000000u

// 64-bit atomics: native on a 64-bit hart, register pairs on a 32-bit hart
// with the AMO64 extension.
#define AMO64_TESTS ((VX_CFG_XLEN == 64) || VX_CFG_EXT_AMO64_ENABLED)

// 64-bit counters start below 2^32 so the low word wraps and the carry into the
// high word is exercised. The add step moves both halves by one, which keeps
// every intermediate value decodable without a division: after k steps the low
// word is base + k and the high word is k, plus one once the low word wrapped.
#define AMO64_ADD_BASE   0x00000000fffffff0ull
#define AMO64_ADD_STEP   0x0000000100000001ull
#define AMO64_CAS_BASE   0x00000000ffffff00ull
#define AMO64_XOR_INIT   0x0123456789abcdefull
#define AMO64_SWAP_INIT  0x5a5a5a5aa5a5a5a5ull
// Marks a group leader's published record in the shared region.
#define AMO64_LMEM_MARK  0x4c4d454d00000000ull

// min/max operands: harts pair up on the high word, so equal high words are
// decided by the low word, and every other pair flips the sign bit, so signed
// and unsigned order disagree. The low word (h + 1) identifies the hart.
static inline uint64_t amo64_minmax_value(uint32_t h) {
  uint32_t g  = h >> 1;
  uint32_t hi = (g * 0x9e3779b1u) ^ ((g & 1) ? 0x80000000u : 0u);
  return ((uint64_t)hi << 32) | (h + 1);
}

// Bit operands: one bit in each half, so a lost half shows in the result.
static inline uint64_t amo64_bit_value(uint32_t h) {
  return (1ull << (h % 32)) | (1ull << (32 + ((h * 7 + 3) % 32)));
}

// Exchange operand: both halves name the hart, so a torn old value is caught.
static inline uint64_t amo64_swap_value(uint32_t h) {
  return ((uint64_t)(h + 1) << 32) | ((h + 1) * 0x01010101u);
}

// The word hart h stores on iteration i of amo64_x0: its low word carries into
// the high one when AMO64_ADD_STEP is added.
static inline uint64_t amo64_x0_value(uint32_t h, uint32_t i) {
  return ((uint64_t)(h + 1 + i) << 32) | (0xfffffff0u + (h & 7));
}

// Compare-and-swap ladder claim; zero is the unclaimed marker.
static inline uint64_t amo64_claim_value(uint32_t h) {
  return ((uint64_t)(h + 1) << 32) | (h + 1);
}

// Args shared between host (main.cpp) and the GPU kernel (kernel.cpp).
// Layout matches dogfood's, with `iters` added for the contention loops.
typedef struct {
  uint32_t testid;
  uint32_t num_harts;     // total harts hammering the shared word
  uint32_t iters;         // per-hart iteration count
  uint64_t shared_addr;   // single-word AMO target (4 or 8 bytes)
  uint64_t per_hart_addr; // per-hart scratch / observed-old buffer
} kernel_arg_t;

#endif
