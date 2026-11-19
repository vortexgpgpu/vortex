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

// Memory pool for fixed-size objects. The pool grows by slabs of PoolSize
// blocks and recycles them; it never falls back to the heap per object.
// Every block carries an owner tag ahead of the payload, so a block may be
// freed on any thread: same-pool frees take the unsynchronized fast path and
// frees of another pool's block go through that pool's lock-free return stack.
template<typename T, size_t PoolSize = 64>
class MemoryPool {
public:
  MemoryPool() {
    this->grow();
  }

  ~MemoryPool() noexcept {
    for (auto* chunk : chunks_) {
      free(chunk);
    }
  }

  T* allocate() {
    if (__builtin_expect(free_list_ == nullptr, 0)) {
      // Reclaim blocks returned by other threads, else add a slab.
      free_list_ = static_cast<char*>(remote_free_.exchange(nullptr, std::memory_order_acquire));
      if (free_list_ == nullptr) {
        this->grow();
      }
    }
    char* block = free_list_;
    free_list_ = next_of(block);
    return payload_of(block);
  }

  void deallocate(T* ptr) noexcept {
    char* block = reinterpret_cast<char*>(ptr) - kHeader;
    auto* owner = owner_of(block);
    if (__builtin_expect(owner == this, 1)) {
      next_of(block) = free_list_;
      free_list_ = block;
    } else {
      owner->remote_free_push(block);
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

  __attribute__((noinline)) void grow() {
    auto chunk = static_cast<char*>(aligned_alloc(kAlign, kStride * PoolSize));
    if (chunk == nullptr) {
      throw std::bad_alloc();
    }
    chunks_.push_back(chunk);
    for (size_t i = 0; i < PoolSize; ++i) {
      char* block = chunk + i * kStride;
      owner_of(block) = this;
      next_of(block) = (i < PoolSize - 1) ? (chunk + (i + 1) * kStride) : free_list_;
    }
    free_list_ = chunk;
  }

  // Push-only Treiber stack; the owner drains it with a single exchange.
  void remote_free_push(char* block) noexcept {
    void* head = remote_free_.load(std::memory_order_relaxed);
    do {
      next_of(block) = static_cast<char*>(head);
    } while (!remote_free_.compare_exchange_weak(head, block,
                 std::memory_order_release, std::memory_order_relaxed));
  }

  std::vector<char*> chunks_;
  char* free_list_ = nullptr;
  std::atomic<void*> remote_free_{nullptr};
};

// Hands every pool a thread bound back to its registry when the thread exits.
// One list serves all pool types: a thread_local holder per type would cost
// static TLS space for each type.
class PoolParking {
public:
  using release_fn = void (*)(void*);

  static void add(release_fn release, void* pool) {
    static thread_local PoolParking parking;
    parking.entries_.push_back({release, pool});
  }

  ~PoolParking() {
    for (auto& e : entries_) {
      e.release(e.pool);
    }
  }

private:
  struct entry_t {
    release_fn release;
    void* pool;
  };
  std::vector<entry_t> entries_;
};

// STL allocator backed by a per-thread MemoryPool.
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

  // Hot path: a trivially-initialized per-thread pointer (no TLS wrapper
  // call); the pool is bound, and its release at thread exit registered, once.
  static MemoryPool<T, PoolSize>& get_pool() {
    static thread_local MemoryPool<T, PoolSize>* cached
        __attribute__((tls_model("initial-exec"))) = nullptr;
    if (__builtin_expect(cached != nullptr, 1)) {
      return *cached;
    }
    cached = bind_pool();
    return *cached;
  }

  __attribute__((noinline)) static MemoryPool<T, PoolSize>* bind_pool() {
    auto* pool = registry().acquire();
    PoolParking::add([](void* p) {
      registry().release(static_cast<MemoryPool<T, PoolSize>*>(p));
    }, pool);
    return pool;
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
