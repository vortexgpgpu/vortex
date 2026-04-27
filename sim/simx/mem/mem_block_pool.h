// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <memory_resource>
#include "types.h"

namespace vortex {

// Pool-backed allocator for cache-line / TLM-payload mem_block_t blocks.
// std::allocate_shared fuses the shared_ptr control block and the payload
// into a single pool allocation, eliminating the malloc/free hot path that
// dominates simulator runtime on memory-heavy workloads.
//
// SimX is single-threaded; unsynchronized_pool_resource is lock-free.
inline std::pmr::unsynchronized_pool_resource& mem_block_pool() {
  static std::pmr::pool_options opts{
    /* max_blocks_per_chunk        */ 1024,
    /* largest_required_pool_block */ sizeof(mem_block_t) + 64
  };
  static std::pmr::unsynchronized_pool_resource pool{opts};
  return pool;
}

inline std::shared_ptr<mem_block_t> make_mem_block() {
  std::pmr::polymorphic_allocator<mem_block_t> alloc(&mem_block_pool());
  return std::allocate_shared<mem_block_t>(alloc);
}

inline std::shared_ptr<mem_block_t> make_mem_block_copy(const mem_block_t& src) {
  std::pmr::polymorphic_allocator<mem_block_t> alloc(&mem_block_pool());
  return std::allocate_shared<mem_block_t>(alloc, src);
}

} // namespace vortex
