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
#include <cassert>

namespace vortex {

// Memory pool for fixed-size objects with fallback to new/delete
template<typename T, size_t PoolSize = 64>
class MemoryPool {
public:
  MemoryPool() {
    // Allocate raw memory without constructing objects
    pool_ = static_cast<char*>(aligned_alloc(alignof(T), sizeof(T) * PoolSize));
    // Initialize free list using pointer arithmetic
    for (size_t i = 0; i < PoolSize; ++i) {
      *reinterpret_cast<void**>(pool_ + i * sizeof(T)) =
        (i < PoolSize - 1) ? pool_ + (i + 1) * sizeof(T) : nullptr;
    }
    free_list_ = pool_;
  }

  ~MemoryPool() noexcept {
    free(pool_);
  }

  T* allocate() {
    if (free_list_) {
      void* block = free_list_;
      free_list_ = *reinterpret_cast<void**>(block);
      return static_cast<T*>(block);
    }
    return static_cast<T*>(::operator new(sizeof(T)));
  }

  void deallocate(T* ptr) noexcept {
    if (belongs_to_pool(ptr)) {
      *reinterpret_cast<void**>(ptr) = free_list_;
      free_list_ = ptr;
    } else {
      ::operator delete(ptr);
    }
  }

private:
  char* pool_ = nullptr;
  void* free_list_ = nullptr;

  bool belongs_to_pool(T* ptr) const noexcept {
    return ptr >= reinterpret_cast<T*>(pool_) &&
           ptr < reinterpret_cast<T*>(pool_ + sizeof(T) * PoolSize);
  }
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

  static MemoryPool<T, PoolSize>& get_pool() {
    static MemoryPool<T, PoolSize> pool;
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