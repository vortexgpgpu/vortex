// Copyright © 2019-2026
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

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#include "util.h"

namespace vortex {

///////////////////////////////////////////////////////////////////////////////
// RingQueue: fixed-capacity FIFO with no dynamic allocations on push/pop.
// Does not require T to be default-constructible. Copy is disabled; use move.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class RingQueue {
public:
  RingQueue() : capacity_(0), head_(0), tail_(0), size_(0) {}

  explicit RingQueue(uint32_t cap)
    : capacity_(cap), head_(0), tail_(0), size_(0) {
    __assert(capacity_ > 0, "RingQueue: capacity must be > 0");
    storage_.reset(new Storage[capacity_]);
  }

  RingQueue(const RingQueue&) = delete;
  RingQueue& operator=(const RingQueue&) = delete;

  RingQueue(RingQueue&& other) noexcept
    : storage_(std::move(other.storage_))
    , capacity_(other.capacity_)
    , head_(other.head_)
    , tail_(other.tail_)
    , size_(other.size_) {
    other.capacity_ = 0;
    other.head_ = other.tail_ = other.size_ = 0;
  }

  RingQueue& operator=(RingQueue&& other) noexcept {
    if (this != &other) {
      this->clear();
      storage_ = std::move(other.storage_);
      capacity_ = other.capacity_;
      head_ = other.head_;
      tail_ = other.tail_;
      size_ = other.size_;
      other.capacity_ = 0;
      other.head_ = other.tail_ = other.size_ = 0;
    }
    return *this;
  }

  ~RingQueue() {
    this->clear();
  }

  bool empty() const { return size_ == 0; }
  bool full() const { return size_ == capacity_; }
  uint32_t size() const { return size_; }
  uint32_t capacity() const { return capacity_; }

  const T& front() const {
    __assert(size_ > 0, "RingQueue: empty");
    return *ptr(head_);
  }

  void pop() {
    __assert(size_ > 0, "RingQueue: empty");
    ptr(head_)->~T();
    head_ = inc(head_);
    --size_;
  }

  void push(const T& v) {
    __assert(size_ < capacity_, "RingQueue: full");
    new (&storage_[tail_]) T(v);
    tail_ = inc(tail_);
    ++size_;
  }

  void push(T&& v) {
    __assert(size_ < capacity_, "RingQueue: full");
    new (&storage_[tail_]) T(std::move(v));
    tail_ = inc(tail_);
    ++size_;
  }

  void clear() {
    while (size_ != 0) {
      this->pop();
    }
  }

  // Double the capacity, keeping the queued elements in order (for queues
  // that are unbounded by design).
  void grow() {
    uint32_t new_cap = (capacity_ != 0) ? capacity_ * 2 : 4;
    std::unique_ptr<Storage[]> grown(new Storage[new_cap]);
    for (uint32_t i = 0, idx = head_; i < size_; ++i, idx = inc(idx)) {
      new (&grown[i]) T(std::move(*ptr(idx)));
      ptr(idx)->~T();
    }
    storage_ = std::move(grown);
    capacity_ = new_cap;
    head_ = 0;
    tail_ = size_;
  }

private:
  using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

  T* ptr(uint32_t idx) const {
    return reinterpret_cast<T*>(&storage_[idx]);
  }

  uint32_t inc(uint32_t idx) const {
    ++idx;
    if (idx == capacity_) idx = 0;
    return idx;
  }

  std::unique_ptr<Storage[]> storage_;
  uint32_t capacity_;
  uint32_t head_;
  uint32_t tail_;
  uint32_t size_;
};

} // namespace vortex
