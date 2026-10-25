#ifndef _COMMON_H_
#define _COMMON_H_

// Test modes, selected by kernel_arg_t::mode.
#define FENCE_MP_PAIRS 0  // mirrored tasks publish to each other, round after round
#define FENCE_MP_LANES 1  // a fence publishes a store another thread of its warp made
#define FENCE_MP_PORTS 2  // a fence publishes stores that left through another memory port

// Lane mode, one warp a block: rounds of loads the upper lanes send ahead of
// the payload store, the k-th of lane l at line (k * threads + l) * 16 so all
// fall in one data-cache bank (up to 16 banks); the payload, the last lane's
// line a round past them; the guard, a round further, with the lines one cache
// way apart that evict it; and the value stored.
#define LANE_THREADS      VX_CFG_NUM_THREADS
#define LANE_FLOOD_ROUNDS 16
#define LANE_BANK_WORDS   (16 * VX_CFG_DCACHE_LINE_SIZE / 4)
#define LANE_ROUND_WORDS  (LANE_THREADS * LANE_BANK_WORDS)
#define LANE_PAYLOAD_WORD (LANE_FLOOD_ROUNDS * LANE_ROUND_WORDS + (LANE_THREADS - 1) * LANE_BANK_WORDS)
#define LANE_GUARD_WORD   ((LANE_FLOOD_ROUNDS + 1) * LANE_ROUND_WORDS)
#define LANE_CACHE_WAYS   VX_CFG_DCACHE_NUM_WAYS
#define LANE_WAY_WORDS    (VX_CFG_DCACHE_SIZE / VX_CFG_DCACHE_NUM_WAYS / 4)
#define LANE_BUFFER_WORDS (LANE_GUARD_WORD + (2 * LANE_CACHE_WAYS + 1) * LANE_WAY_WORDS)
#define LANE_PAYLOAD      0x5A5A0001u

// Port mode, one warp a block, for a data cache with two banks, each on its
// own memory port, below which a cache of lines twice as long has two banks:
// line k of (port, bank) sets the data-cache bank bit to the port and the
// next-level bank bit to the bank. The writer stores PORT_STORE_ROUNDS lines
// a thread, then the payload, once all blocks have had
// PORT_WARMUP_CYCLES to start; the other blocks load PORT_FLOOD_ROUNDS lines
// a thread from the line PORT_FLOOD_LINE on.
#define PORT_THREADS       VX_CFG_NUM_THREADS
#define PORT_LINE_OFFSET(port, bank, k) \
  ((k) * 2 * VX_CFG_L2_LINE_SIZE + (bank) * VX_CFG_L2_LINE_SIZE + (port) * VX_CFG_DCACHE_LINE_SIZE)
#define PORT_STORE_ROUNDS  4
#define PORT_FLOOD_LINE    (PORT_STORE_ROUNDS * PORT_THREADS)
#define PORT_FLOOD_ROUNDS  128
#define PORT_WARMUP_CYCLES 2000
#define PORT_BUFFER_WORDS(blocks) \
  (PORT_LINE_OFFSET(0, 0, PORT_FLOOD_LINE + ((blocks) - 2) * PORT_THREADS * PORT_FLOOD_ROUNDS) / 4)
#define PORT_PAYLOAD       0x5A5A0100u

typedef struct {
  uint32_t num_tasks;
  uint32_t rounds;
  uint32_t plain_guard;
  uint32_t mode;
  uint64_t payload_addr;
  uint64_t guard_addr;
  uint64_t result_addr;
  uint64_t flood_addr;
} kernel_arg_t;

// The payload word a task publishes in a round: nonzero, so a payload still
// at its cleared value never matches.
static inline uint32_t payload_value(uint32_t round, uint32_t task) {
  return (round << 20) + task + 1;
}

#endif
