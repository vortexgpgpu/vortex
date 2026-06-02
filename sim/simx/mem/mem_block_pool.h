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
// Lifetime: the pool is owned by a shared_ptr held inside every block's
// allocator copy. Each live mem_block_t pins the pool alive; the pool is
// destroyed when the static singleton drops its reference AND no live
// blocks remain — destruction order between libsimx.so and the runtime
// does not matter.
//
// SimX is single-threaded; unsynchronized_pool_resource is lock-free.

namespace detail {

struct mem_block_pool_t : public std::pmr::memory_resource {
  std::pmr::unsynchronized_pool_resource pool;

  explicit mem_block_pool_t(const std::pmr::pool_options& opts) : pool(opts) {}

  void* do_allocate(std::size_t bytes, std::size_t align) override {
    return pool.allocate(bytes, align);
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t align) override {
    pool.deallocate(p, bytes, align);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

// Allocator that owns its pool via shared_ptr. allocate_shared copies this
// into the control block, so the pool stays alive for as long as any block
// (or the static singleton in mem_block_pool()) holds a reference.
template <typename T>
class pool_allocator {
public:
  using value_type = T;
  template <typename U> friend class pool_allocator;

  explicit pool_allocator(std::shared_ptr<mem_block_pool_t> pool)
    : pool_(std::move(pool)) {}

  template <typename U>
  pool_allocator(const pool_allocator<U>& other) noexcept
    : pool_(other.pool_) {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(pool_->allocate(n * sizeof(T), alignof(T)));
  }
  void deallocate(T* p, std::size_t n) noexcept {
    pool_->deallocate(p, n * sizeof(T), alignof(T));
  }

  template <typename U>
  bool operator==(const pool_allocator<U>& other) const noexcept {
    return pool_ == other.pool_;
  }
  template <typename U>
  bool operator!=(const pool_allocator<U>& other) const noexcept {
    return !(*this == other);
  }

private:
  std::shared_ptr<mem_block_pool_t> pool_;
};

inline const std::shared_ptr<mem_block_pool_t>& mem_block_pool_singleton() {
  static const auto pool = std::make_shared<mem_block_pool_t>(
    std::pmr::pool_options{
      /* max_blocks_per_chunk        */ 1024,
      /* largest_required_pool_block */ sizeof(mem_block_t) + 64
    });
  return pool;
}

} // namespace detail

inline std::shared_ptr<mem_block_t> make_mem_block() {
  detail::pool_allocator<mem_block_t> alloc(detail::mem_block_pool_singleton());
  return std::allocate_shared<mem_block_t>(alloc);
}

inline std::shared_ptr<mem_block_t> make_mem_block_copy(const mem_block_t& src) {
  detail::pool_allocator<mem_block_t> alloc(detail::mem_block_pool_singleton());
  return std::allocate_shared<mem_block_t>(alloc, src);
}

} // namespace vortex
