#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Every access goes through volatile pointers: the point is to send each load
// to the cache right behind the stores it must observe, not to let the
// compiler forward the stored value.
static inline volatile uint32_t* region_ptr(const kernel_arg_t* arg, uint32_t region) {
  return reinterpret_cast<volatile uint32_t*>(arg->buf_addr) + region * REGION_WORDS;
}

static inline uint32_t check_word(volatile uint32_t* r, uint32_t region, uint32_t w,
                                  const kernel_arg_t* arg) {
  uint32_t ref = expected_word(region, w, arg);
  return (r[w] != ref) ? 1 : 0;
}

static uint32_t run_half(const kernel_arg_t* arg, uint32_t region) {
  auto r = region_ptr(arg, region);
  for (uint32_t w = 0; w < REGION_WORDS / 2; ++w) {
    r[w] = store_value(region, w);
  }
  uint32_t errors = 0;
  for (uint32_t w = 0; w < REGION_WORDS; ++w) {
    errors += check_word(r, region, w, arg);
  }
  return errors;
}

static uint32_t run_bytes(const kernel_arg_t* arg, uint32_t region) {
  auto r = region_ptr(arg, region);
  reinterpret_cast<volatile uint8_t*>(r)[5] = 0xee;
  uint32_t errors = check_word(r, region, 1, arg);
  reinterpret_cast<volatile uint16_t*>(r)[20] = 0xbeef;
  errors += check_word(r, region, 10, arg);
  errors += check_word(r, region, 0, arg);
  errors += check_word(r, region, 15, arg);
  return errors;
}

static uint32_t run_amo(const kernel_arg_t* arg, uint32_t region) {
  auto r = region_ptr(arg, region);
  for (uint32_t w = 0; w < 4; ++w) {
    r[w] = store_value(region, w);
  }
  if (!arg->amo) {
    return check_word(r, region, 8, arg) + check_word(r, region, 1, arg);
  }
  uint32_t errors = 0;
  uint32_t* word8 = const_cast<uint32_t*>(&r[8]);
  uint32_t* word1 = const_cast<uint32_t*>(&r[1]);
  uint32_t* word12 = const_cast<uint32_t*>(&r[12]);
  uint32_t old8 = __atomic_fetch_add(word8, 5u, __ATOMIC_SEQ_CST);
  errors += (old8 != pattern_word(region * REGION_WORDS + 8)) ? 1 : 0;
  uint32_t old1 = __atomic_fetch_add(word1, 3u, __ATOMIC_SEQ_CST);
  errors += (old1 != store_value(region, 1)) ? 1 : 0;
  // Only this thread touches the word, so the SC fails only spuriously.
  uint32_t first = 0;
  bool done = false;
  for (uint32_t attempt = 0; attempt < 64 && !done; ++attempt) {
    uint32_t old, sc_fail;
    asm volatile ("lr.w %0, (%1)" : "=r"(old) : "r"(word12) : "memory");
    if (attempt == 0) {
      first = old;
    }
    uint32_t newv = old + 1;
    asm volatile ("sc.w %0, %2, (%1)" : "=r"(sc_fail) : "r"(word12), "r"(newv) : "memory");
    done = (sc_fail == 0);
  }
  errors += (!done || first != pattern_word(region * REGION_WORDS + 12)) ? 1 : 0;
  return errors;
}

static void run_store(const kernel_arg_t* arg, uint32_t region) {
  auto b = reinterpret_cast<volatile uint8_t*>(region_ptr(arg, region));
  for (uint32_t w = 0; w < REGION_WORDS; w += 2) {
    b[w * 4] = (uint8_t)(0x30 + w);
  }
}

// Each lane's load follows the warp's coalesced store into the same line
// right away, so it reaches the cache while the store's allocation is still
// pending and must wait for the unwritten half to be fetched.
static uint32_t run_lane(const kernel_arg_t* arg, uint32_t round, uint32_t gid) {
  uint32_t lane  = gid % arg->warp_size;
  uint32_t lanes = (arg->warp_size < LANE_WORDS) ? arg->warp_size : LANE_WORDS;
  if (lane >= lanes) {
    return 0;
  }
  uint32_t region = region_index(round, arg->num_threads, gid - lane, CASE_LANE);
  auto r = region_ptr(arg, region);
  uint32_t value = store_value(region, lane);
  uint32_t loaded;
  asm volatile ("sw %2, 0(%1)\n\tlw %0, %3(%1)"
                : "=&r"(loaded)
                : "r"(&r[lane]), "r"(value), "i"(LANE_WORDS * 4)
                : "memory");
  uint32_t errors = (loaded != expected_word(region, lane + LANE_WORDS, arg)) ? 1 : 0;
  errors += check_word(r, region, (lane + 1) % lanes, arg);
  return errors;
}

// Store word 0 of this thread's same-set line, then load word 8 right away.
static uint32_t run_set(const kernel_arg_t* arg, uint32_t round, uint32_t gid) {
  uint32_t line = set_line(round, arg->num_threads, gid);
  auto r = reinterpret_cast<volatile uint32_t*>(arg->set_addr) + line * SET_STRIDE_WORDS;
  uint32_t value = expected_set_word(line * SET_STRIDE_WORDS);
  uint32_t loaded;
  asm volatile ("sw %2, 0(%1)\n\tlw %0, 32(%1)"
                : "=&r"(loaded)
                : "r"(r), "r"(value)
                : "memory");
  uint32_t errors = (loaded != expected_set_word(line * SET_STRIDE_WORDS + 8)) ? 1 : 0;
  errors += (r[0] != value) ? 1 : 0;
  return errors;
}

static uint32_t check_region(const kernel_arg_t* arg, uint32_t region) {
  auto r = region_ptr(arg, region);
  uint32_t errors = 0;
  for (uint32_t w = 0; w < REGION_WORDS; ++w) {
    errors += check_word(r, region, w, arg);
  }
  return errors;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t errors = 0;
  auto err = reinterpret_cast<volatile uint32_t*>(arg->err_addr);

  if (arg->phase == 3) {
    // Read fresh lines from memory, one word each.
    auto base = reinterpret_cast<volatile uint32_t*>(arg->full_addr);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < arg->full_lines; ++i) {
      uint32_t line = i * arg->num_threads + gid;
      sum += base[line * (FLOOD_LINE_BYTES / sizeof(uint32_t))];
    }
    asm volatile ("" :: "r"(sum));
    err[gid] = 0;
    return;
  }

  if (arg->phase == 2 && arg->full_mode == FULL_CHAIN) {
    // The first SECTOR_WORDS threads of a block store one word each of a
    // sector, every warp at once after a barrier, and load it right away: the
    // loads reach the cache while the stores' allocation is still pending.
    // The second store lands while that chain is being refetched.
    auto base = reinterpret_cast<volatile uint32_t*>(arg->full_addr);
    uint32_t t = threadIdx.x;
    for (uint32_t i = 0; i < arg->full_lines; ++i) {
      uint32_t line = i * arg->num_blocks + blockIdx.x;
      for (uint32_t s = 0; s < FULL_LINE_WORDS; s += SECTOR_WORDS) {
        uint32_t idx = line * FULL_LINE_WORDS + s + t;
        bool skip = full_skipped(line, s + t, arg);
        // a skipped word is stored to a scratch word: no branch delays the load
        volatile uint32_t* dst = skip ? &err[arg->num_threads + gid] : &base[idx];
        __syncthreads();
        if (t < SECTOR_WORDS) {
          *dst = full_value(line, s + t);
          uint32_t first = base[idx];
          *dst = full_value2(line, s + t);
          uint32_t second = base[idx];
          errors += (first != (skip ? pattern_word(idx) : full_value(line, s + t))) ? 1 : 0;
          errors += (second != full_expected(idx, arg)) ? 1 : 0;
        }
      }
    }
    err[gid] = errors;
    return;
  }

  if (arg->phase == 2) {
    // Store this thread's lines, all of each sector or all but one word,
    // then load them all back.
    auto base = reinterpret_cast<volatile uint32_t*>(arg->full_addr);
    for (uint32_t i = 0; i < arg->full_lines; ++i) {
      uint32_t line = i * arg->num_threads + gid;
      for (uint32_t w = 0; w < FULL_LINE_WORDS; ++w) {
        if (!full_skipped(line, w, arg)) {
          base[line * FULL_LINE_WORDS + w] = full_value(line, w);
        }
      }
    }
    for (uint32_t i = 0; i < arg->full_lines; ++i) {
      uint32_t line = i * arg->num_threads + gid;
      for (uint32_t w = 0; w < FULL_LINE_WORDS; ++w) {
        uint32_t idx = line * FULL_LINE_WORDS + w;
        errors += (base[idx] != full_expected(idx, arg)) ? 1 : 0;
      }
    }
    err[gid] = errors;
    return;
  }

  if (arg->phase != 0) {
    // Every region again, from another thread (another core when there are
    // several), after the launch boundary flushed the partial lines.
    uint32_t owner = (gid + arg->rotate) % arg->num_threads;
    for (uint32_t round = 0; round < arg->rounds; ++round) {
      for (uint32_t c = 0; c < NUM_CASES; ++c) {
        errors += check_region(arg, region_index(round, arg->num_threads, owner, c));
      }
      auto r = reinterpret_cast<volatile uint32_t*>(arg->set_addr);
      uint32_t base = set_line(round, arg->num_threads, owner) * SET_STRIDE_WORDS;
      for (uint32_t w = 0; w < REGION_WORDS; ++w) {
        errors += (r[base + w] != expected_set_word(base + w)) ? 1 : 0;
      }
    }
    err[arg->num_threads + gid] = errors;
    return;
  }

  for (uint32_t round = 0; round < arg->rounds; ++round) {
    errors += run_half(arg, region_index(round, arg->num_threads, gid, CASE_HALF));
    errors += run_bytes(arg, region_index(round, arg->num_threads, gid, CASE_BYTES));
    errors += run_amo(arg, region_index(round, arg->num_threads, gid, CASE_AMO));
    run_store(arg, region_index(round, arg->num_threads, gid, CASE_STORE));
    errors += run_lane(arg, round, gid);
    errors += run_set(arg, round, gid);
    if (arg->cross_warp) {
      bool producer = warp_producer(gid, arg->warp_size);
      if (producer) {
        uint32_t region = region_index(round, arg->num_threads, gid, CASE_WARP);
        auto r = region_ptr(arg, region);
        for (uint32_t w = REGION_WORDS / 2; w < REGION_WORDS; ++w) {
          r[w] = store_value(region, w);
        }
      }
      vx_fence();
      __syncthreads();
      if (!producer) {
        errors += check_region(arg, region_index(round, arg->num_threads, gid - arg->warp_size, CASE_WARP));
      }
    }
  }
  err[gid] = errors;
}
