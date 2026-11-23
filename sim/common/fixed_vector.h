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

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <util.h>

namespace vortex {

// A vector with inline storage whose capacity is a compile-time bound (a
// VX_CFG_* thread or operand count): the std::vector subset the simulator
// uses, without a heap allocation per object or per copy. The size is still
// dynamic up to the capacity; exceeding it is an assertion failure.
template <typename T, uint32_t N>
class FixedVector {
public:
  using value_type      = T;
  using size_type       = uint32_t;
  using iterator        = T*;
  using const_iterator  = const T*;
  using reference       = T&;
  using const_reference = const T&;

  FixedVector() : size_(0) {}

  explicit FixedVector(size_type n) : size_(0) {
    this->assign(n, T{});
  }

  FixedVector(size_type n, const T& value) : size_(0) {
    this->assign(n, value);
  }

  FixedVector(std::initializer_list<T> init) : size_(0) {
    __assert(init.size() <= N, "FixedVector: capacity " << N << " exceeded");
    for (auto& v : init) {
      data_[size_++] = v;
    }
  }

  static constexpr size_type capacity() { return N; }

  size_type size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  iterator begin() { return data_.data(); }
  iterator end() { return data_.data() + size_; }
  const_iterator begin() const { return data_.data(); }
  const_iterator end() const { return data_.data() + size_; }

  reference operator[](size_type i) { return data_[i]; }
  const_reference operator[](size_type i) const { return data_[i]; }

  reference at(size_type i) {
    __assert(i < size_, "FixedVector: index " << i << " out of range " << size_);
    return data_[i];
  }
  const_reference at(size_type i) const {
    __assert(i < size_, "FixedVector: index " << i << " out of range " << size_);
    return data_[i];
  }

  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }
  reference back() { return data_[size_ - 1]; }
  const_reference back() const { return data_[size_ - 1]; }

  void assign(size_type n, const T& value) {
    __assert(n <= N, "FixedVector: capacity " << N << " exceeded");
    for (size_type i = 0; i < n; ++i) {
      data_[i] = value;
    }
    size_ = n;
  }

  void resize(size_type n, const T& value = T{}) {
    __assert(n <= N, "FixedVector: capacity " << N << " exceeded");
    for (size_type i = size_; i < n; ++i) {
      data_[i] = value;
    }
    size_ = n;
  }

  void push_back(const T& value) {
    __assert(size_ < N, "FixedVector: capacity " << N << " exceeded");
    data_[size_++] = value;
  }

  void pop_back() { --size_; }
  void clear() { size_ = 0; }

private:
  std::array<T, N> data_;
  size_type size_;
};

} // namespace vortex
