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

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace vortex {

// Memory pool for fixed-size objects with fallback to new/delete.
// Every block carries an owner tag ahead of the payload, so a block may be
// freed on any thread: same-pool frees take the unsynchronized fast path,
// frees of another pool's block go through that pool's lock-free return
// stack, and heap-fallback blocks are deleted directly.
template<typename T, size_t PoolSize = 64>
class MemoryPool {
public:
  MemoryPool() {
    chunk_ = static_cast<char*>(aligned_alloc(kAlign, kStride * PoolSize));
    for (size_t i = 0; i < PoolSize; ++i) {
      char* block = chunk_ + i * kStride;
      owner_of(block) = this;
      next_of(block) = (i < PoolSize - 1) ? (chunk_ + (i + 1) * kStride) : nullptr;
    }
    free_list_ = chunk_;
  }

  ~MemoryPool() noexcept {
    free(chunk_);
  }

  T* allocate() {
    if (!free_list_) {
      // Reclaim blocks returned by other threads.
      free_list_ = static_cast<char*>(remote_free_.exchange(nullptr, std::memory_order_acquire));
    }
    if (free_list_) {
      char* block = free_list_;
      free_list_ = next_of(block);
      return payload_of(block);
    }
    char* block = static_cast<char*>(::operator new(kStride, std::align_val_t(kAlign)));
    owner_of(block) = nullptr;
    return payload_of(block);
  }

  void deallocate(T* ptr) noexcept {
    char* block = reinterpret_cast<char*>(ptr) - kHeader;
    auto* owner = owner_of(block);
    if (owner == this) {
      next_of(block) = free_list_;
      free_list_ = block;
    } else if (owner) {
      owner->remote_free_push(block);
    } else {
      ::operator delete(block, std::align_val_t(kAlign));
    }
  }

private:
  static constexpr size_t kAlign  = (alignof(T) > alignof(void*)) ? alignof(T) : alignof(void*);
  static constexpr size_t kHeader = kAlign;  // owner tag, padded to payload alignment
  static constexpr size_t kStride = kHeader + ((sizeof(T) + kAlign - 1) / kAlign) * kAlign;

  // The free-list link is stored in the (dead) payload area.
  static_assert(sizeof(T) >= sizeof(void*), "pooled type too small");

  static MemoryPool*& owner_of(char* block) noexcept {
    return *reinterpret_cast<MemoryPool**>(block);
  }
  static char*& next_of(char* block) noexcept {
    return *reinterpret_cast<char**>(block + kHeader);
  }
  static T* payload_of(char* block) noexcept {
    return reinterpret_cast<T*>(block + kHeader);
  }

  // Push-only Treiber stack; the owner drains it with a single exchange.
  void remote_free_push(char* block) noexcept {
    void* head = remote_free_.load(std::memory_order_relaxed);
    do {
      next_of(block) = static_cast<char*>(head);
    } while (!remote_free_.compare_exchange_weak(head, block,
                 std::memory_order_release, std::memory_order_relaxed));
  }

  char* chunk_ = nullptr;
  char* free_list_ = nullptr;
  std::atomic<void*> remote_free_{nullptr};
};

// Custom allocator using the memory pool with fallback
template <typename T, size_t PoolSize = 64>
class PoolAllocator {
public:
  using value_type = T;

  PoolAllocator() = default;

  template <typename U>
  PoolAllocator(const PoolAllocator<U, PoolSize>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n != 1) throw std::bad_alloc();
    return get_pool().allocate();
  }

  void deallocate(T* p, std::size_t n) noexcept {
    if (n == 1) get_pool().deallocate(p);
  }

  template<typename U>
  struct rebind {
    using other = PoolAllocator<U, PoolSize>;
  };

  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

private:
  template<typename, size_t> friend class PoolAllocator;

  // Pools are per-thread so the hot path needs no locking, but a block may be
  // freed on a different thread than allocated it (owner-tag return path) and
  // may outlive its allocating thread. An exiting thread parks its pool in a
  // process-lifetime registry for a later thread to adopt; pools are never
  // destroyed while their blocks can still be in flight.
  struct registry_t {
    std::mutex lock;
    std::vector<MemoryPool<T, PoolSize>*> idle;

    MemoryPool<T, PoolSize>* acquire() {
      std::lock_guard<std::mutex> g(lock);
      if (!idle.empty()) {
        auto* pool = idle.back();
        idle.pop_back();
        return pool;
      }
      return new MemoryPool<T, PoolSize>();
    }
    void release(MemoryPool<T, PoolSize>* pool) {
      std::lock_guard<std::mutex> g(lock);
      idle.push_back(pool);
    }
  };

  static registry_t& registry() {
    static registry_t* registry = new registry_t();  // immortal: outlives every thread
    return *registry;
  }

  static MemoryPool<T, PoolSize>& get_pool() {
    struct holder_t {
      MemoryPool<T, PoolSize>* pool;
      holder_t() : pool(registry().acquire()) {}
      ~holder_t() { registry().release(pool); }
    };
    static thread_local holder_t holder;
    return *holder.pool;
  }
};

// Comparisons required by STL containers
template<typename T1, typename T2, size_t N>
bool operator==(const PoolAllocator<T1, N>&, const PoolAllocator<T2, N>&) noexcept {
  return true;
}

template<typename T1, typename T2, size_t N>
bool operator!=(const PoolAllocator<T1, N>&, const PoolAllocator<T2, N>&) noexcept {
  return false;
}

}
