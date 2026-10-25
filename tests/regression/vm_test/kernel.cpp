#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <VX_types.h>
#include <type_traits>
#include "common.h"

#define PAGE_SIZE 4096u
#define WORDS_PER_PAGE (PAGE_SIZE / sizeof(uint32_t))

// Page-table entries are XLEN-shaped: 4 bytes under Sv32, 8 under Sv39.
using pte_word_t = typename std::conditional<VX_VM_PTE_SIZE == 4,
                                             uint32_t, uint64_t>::type;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= arg->num_tasks) {
    return;
  }

  auto buf = reinterpret_cast<volatile uint32_t*>(arg->buf_addr);
  auto dst = reinterpret_cast<volatile uint32_t*>(arg->dst_addr);
  uint32_t pages = arg->pages_per_task;
  uint32_t base_page = gid * pages;

  switch (arg->mode) {
  case VM_MODE_STRIDE: {
    // One write + one read per page: every page is a distinct VPN, so
    // the sweep exercises fill, reach, and eviction at each TLB level
    // depending on the configured footprint.
    uint32_t sum = 0;
    for (uint32_t p = 0; p < pages; ++p) {
      uint32_t idx = (base_page + p) * WORDS_PER_PAGE;
      buf[idx] = gid ^ (0x1234567u + p);
      sum += buf[idx];
    }
    dst[gid] = sum;
    break;
  }
  case VM_MODE_FENCE: {
    // Store to a cold page, fence, barrier, then read the neighbor's
    // page: the fence must not overtake the parked store.
    uint32_t idx = base_page * WORDS_PER_PAGE;
    buf[idx] = 0xAB000000u + gid;
    vx_fence();
    __syncthreads();
    uint32_t peer = (gid + 1) % arg->num_tasks;
    dst[gid] = buf[peer * pages * WORDS_PER_PAGE];
    break;
  }
  case VM_MODE_RELEASE: {
    // Release and acquire between block 0, the writer, and block 1, whose
    // first thread reads; the blocks run in different clusters. The writer
    // stores to its page, fences, and sets the guard with an atomic; the
    // reader waits for the guard, fences, and reads the page. The guard page
    // is translated on both sides up front, and the page on the reader's
    // side. The writer's lanes first store to cold pages, one per page-table
    // region, and the page store goes out in the last burst, so it parks on a
    // TLB miss behind all their walks while the guard swap, a TLB hit, could
    // overtake it. No barrier or kernel end comes in between: the fence alone
    // must keep the parked store ahead of the guard.
    auto guard = reinterpret_cast<uint32_t*>(arg->aux_addr);
    volatile uint32_t* page = buf;
    uint32_t lane = threadIdx.x;
    (void)__atomic_fetch_add(&guard[0], 0u, __ATOMIC_RELAXED);
    if (blockIdx.x == 0) {
      vx_fence();  // the guard translation is in before the cold stores go out
      #pragma unroll
      for (uint32_t r = 0; r < RELEASE_PAD_BURSTS; ++r) {
        uint32_t region = 1 + r * RELEASE_BLOCK_SIZE + lane;
        page[region * VM_REGION_PAGES * WORDS_PER_PAGE] = region;
      }
      uint32_t region = (lane == 0) ? 0 : (RELEASE_PAD_BURSTS * RELEASE_BLOCK_SIZE + lane);
      page[region * VM_REGION_PAGES * WORDS_PER_PAGE] = (lane == 0) ? 0xAB000000u : region;
      vx_fence();
      if (lane == 0) {
        __atomic_exchange_n(&guard[0], 1u, __ATOMIC_RELAXED);
      }
      dst[gid] = 0xAB000000u;
    } else if (blockIdx.x == 1 && lane == 0) {
      (void)page[0];
      uint32_t seen = 0;
      for (uint32_t i = 0; i < 65536 && !seen; ++i) {
        seen = __atomic_fetch_add(&guard[0], 0u, __ATOMIC_RELAXED);
      }
      vx_fence();
      dst[gid] = seen ? page[0] : 0u;
    }
    break;
  }
  case VM_MODE_DRAIN: {
    // Trailing store burst to fresh pages with no fence: completion
    // must not be signaled while any store is still parked on a miss.
    for (uint32_t p = 0; p < pages; ++p) {
      uint32_t idx = (base_page + p) * WORDS_PER_PAGE;
      buf[idx] = 0xD0000000u + gid * pages + p;
    }
    dst[gid] = gid;
    break;
  }
  case VM_MODE_AMO: {
    // All threads accumulate into a small set of counters on distinct
    // cold pages; atomics carry write intent with rw=0.
    auto ctrs = reinterpret_cast<uint32_t*>(arg->aux_addr);
    for (uint32_t p = 0; p < pages; ++p) {
      __atomic_fetch_add(&ctrs[p * WORDS_PER_PAGE], gid + 1, __ATOMIC_RELAXED);
    }
    dst[gid] = gid;
    break;
  }
  case VM_MODE_SUPERPAGE: {
    // The page-table region is identity-mapped with superpage leaves, so
    // reading its own top-level PTE both exercises the megapage entry and
    // states what that entry must be: a valid leaf mapping the region to
    // itself. A wrong or missing megapage translation reads a different
    // physical page and the check fails.
    constexpr uint64_t PT_FANOUT = VX_VM_PT_SIZE / VX_VM_PTE_SIZE;
    constexpr uint64_t VA = VX_MEM_PAGE_TABLE_BASE_ADDR;
    constexpr uint64_t PPN_MASK = VX_VM_PT_LEVEL == 2 ? 0x3FFFFFull : 0xFFFFFFFFFFFull;
    uint32_t level_bits = 0;
    for (uint64_t f = PT_FANOUT; f > 1; f >>= 1) {
      ++level_bits;
    }

    uint64_t ppn = csr_read(VX_CSR_SATP) & PPN_MASK;
    int level = VX_VM_PT_LEVEL - 1;
    uint64_t pte = 0;
    for (; level >= 0; --level) {
      uint64_t idx = (VA >> (VX_VM_PAGE_LOG2_SIZE + level * level_bits)) & (PT_FANOUT - 1);
      auto slot = reinterpret_cast<volatile pte_word_t*>(
          ppn * VX_VM_PT_SIZE + idx * VX_VM_PTE_SIZE);
      pte = *slot;
      if ((pte & 0x1u) == 0 || (pte & 0xEu) != 0) {
        break;                                   // invalid, or a leaf
      }
      ppn = (pte >> 10) & PPN_MASK;
    }

    bool valid = (pte & 0x1u) != 0;              // V
    bool leaf  = (pte & 0xEu) != 0;              // R | W | X
    // The region is mapped with superpages, so the walk must stop above
    // level 0, and the leaf must map the region onto itself.
    bool superpage = level > 0;
    uint64_t span_pages = superpage ? (1ull << (level * level_bits)) : 1;
    bool identity = ((pte >> 10) & PPN_MASK)
                 == ((VA >> VX_VM_PAGE_LOG2_SIZE) & ~(span_pages - 1));
    dst[gid] = (valid && leaf && superpage && identity) ? 1u : 0u;
    break;
  }
  default:
    dst[gid] = 0xDEADu;
    break;
  }
}
