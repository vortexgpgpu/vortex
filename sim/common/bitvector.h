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

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace vortex {

template <typename T = uint32_t>
class BitVector {
private:
  static constexpr size_t BITS_PER_WORD = sizeof(T) * 8;
  std::vector<T> bits_;
  size_t size_;
  bool all_zero_;

  size_t wordIndex(size_t pos) const {
    return pos / BITS_PER_WORD;
  }

  T bitMask(size_t pos) const {
    return T(1) << (pos % BITS_PER_WORD);
  }

  void updateAllZero() {
    all_zero_ = std::all_of(bits_.begin(), bits_.end(), [](T word) { return word == 0; });
  }

public:
  explicit BitVector(size_t size = 0)
    : bits_((size + (BITS_PER_WORD - 1)) / BITS_PER_WORD)
    , size_(size)
    , all_zero_(true)
  {}

  void set(size_t pos) {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    bits_[this->wordIndex(pos)] |= this->bitMask(pos);
    all_zero_ = false;
  }

  void set(size_t pos, bool value) {
    if (value) {
      this->set(pos);
    } else {
      this->reset(pos);
    }
  }

  void reset() {
    std::fill(bits_.begin(), bits_.end(), 0);
    all_zero_ = true;
  }

  void reset(size_t pos) {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    bits_[this->wordIndex(pos)] &= ~this->bitMask(pos);
    this->updateAllZero();
  }

  bool test(size_t pos) const {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    return bits_[this->wordIndex(pos)] & this->bitMask(pos);
  }

  size_t size() const {
    return size_;
  }

  void resize(size_t new_size) {
    size_ = new_size;
    bits_.resize((new_size + (BITS_PER_WORD - 1)) / BITS_PER_WORD, 0);
    this->updateAllZero();
  }

  bool operator==(const BitVector& other) const {
    return (size_ == other.size_) && (bits_ == other.bits_);
  }

  bool operator!=(const BitVector& other) const {
    return !(*this == other);
  }

  bool operator[](size_t pos) const {
    return test(pos);
  }

  BitVector& operator&=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    for (size_t i = 0; i < bits_.size(); ++i) {
      bits_[i] &= other.bits_[i];
    }
    this->updateAllZero();
    return *this;
  }

  BitVector& operator|=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    for (size_t i = 0; i < bits_.size(); ++i) {
      bits_[i] |= other.bits_[i];
    }
    this->updateAllZero();
    return *this;
  }

  BitVector& operator^=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    for (size_t i = 0; i < bits_.size(); ++i) {
      bits_[i] ^= other.bits_[i];
    }
    this->updateAllZero();
    return *this;
  }

  BitVector operator~() const {
    BitVector result(size_);
    for (size_t i = 0; i < bits_.size(); ++i) {
      result.bits_[i] = ~bits_[i];
    }
    result.updateAllZero();
    return result;
  }

  void flip() {
    for (auto &word : bits_) {
      word = ~word;
    }
    this->updateAllZero();
  }

  size_t count() const {
    size_t count = 0;
    for (const auto &word : bits_) {
      count += std::bitset<BITS_PER_WORD>(word).count();
    }
    return count;
  }

  bool none() const {
    return all_zero_;
  }

  bool any() const {
    return !all_zero_;
  }

  bool all() const {
    size_t full_bits = size_ / BITS_PER_WORD;
    size_t remaining_bits = size_ % BITS_PER_WORD;
    T full_mask = ~T(0);
    for (size_t i = 0; i < full_bits; ++i) {
      if (bits_[i] != full_mask)
        return false;
    }
    if (remaining_bits > 0) {
      T partial_mask = (T(1) << remaining_bits) - 1;
      if ((bits_[full_bits] & partial_mask) != partial_mask)
        return false;
    }
    return true;
  }

   BitVector& operator<<=(size_t pos) {
    if (pos >= size_) {
      reset();
      return *this;
    }

    size_t word_shift = pos / BITS_PER_WORD;
    size_t bit_shift = pos % BITS_PER_WORD;

    if (word_shift > 0) {
      for (size_t i = bits_.size() - 1; i >= word_shift; --i) {
        bits_[i] = bits_[i - word_shift];
      }
      std::fill(bits_.begin(), bits_.begin() + word_shift, 0);
    }

    if (bit_shift > 0) {
      for (size_t i = bits_.size() - 1; i > 0; --i) {
        bits_[i] = (bits_[i] << bit_shift) | (bits_[i - 1] >> (BITS_PER_WORD - bit_shift));
      }
      bits_[0] <<= bit_shift;
    }

    this->updateAllZero();
    return *this;
  }

  BitVector& operator>>=(size_t pos) {
    if (pos >= size_) {
      reset();
      return *this;
    }

    size_t word_shift = pos / BITS_PER_WORD;
    size_t bit_shift = pos % BITS_PER_WORD;

    if (word_shift > 0) {
      for (size_t i = 0; i < bits_.size() - word_shift; ++i) {
        bits_[i] = bits_[i + word_shift];
      }
      std::fill(bits_.end() - word_shift, bits_.end(), 0);
    }

    if (bit_shift > 0) {
      for (size_t i = 0; i < bits_.size() - 1; ++i) {
        bits_[i] = (bits_[i] >> bit_shift) | (bits_[i + 1] << (BITS_PER_WORD - bit_shift));
      }
      bits_.back() >>= bit_shift;
    }

    this->updateAllZero();
    return *this;
  }

  std::string to_string() const {
    std::string result;
    for (size_t i = 0; i < size_; ++i) {
      result.push_back(test(i) ? '1' : '0');
    }
    return result;
  }

  unsigned long to_ulong() const {
    if (size_ > sizeof(unsigned long) * 8) {
      throw std::overflow_error("BitVector size exceeds unsigned long capacity");
    }

    unsigned long result = 0;
    for (size_t i = 0; i < size_; ++i) {
      if (test(i)) {
        result |= (1UL << i);
      }
    }
    return result;
  }

  unsigned long long to_ullong() const {
    if (size_ > sizeof(unsigned long long) * 8) {
      throw std::overflow_error("BitVector size exceeds unsigned long long capacity");
    }

    unsigned long long result = 0;
    for (size_t i = 0; i < size_; ++i) {
      if (test(i)) {
        result |= (1ULL << i);
      }
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const BitVector& bv) {
    for (size_t i = 0; i < bv.size_; ++i) {
      os << bv.test(i);
    }
    return os;
  }

  friend BitVector operator&(const BitVector& lhs, const BitVector& rhs) {
    BitVector result(lhs);
    result &= rhs;
    return result;
  }

  friend BitVector operator|(const BitVector& lhs, const BitVector& rhs) {
    BitVector result(lhs);
    result |= rhs;
    return result;
  }

  friend BitVector operator^(const BitVector& lhs, const BitVector& rhs) {
    BitVector result(lhs);
    result ^= rhs;
    return result;
  }

  friend BitVector operator<<(const BitVector& lhs, size_t pos) {
    BitVector result(lhs);
    result <<= pos;
    return result;
  }

  friend BitVector operator>>(const BitVector& lhs, size_t pos) {
    BitVector result(lhs);
    result >>= pos;
    return result;
  }
};

}

// std::hash specialization for BitVector
namespace std {

template <typename T>
struct hash<vortex::BitVector<T>> {
  size_t operator()(const vortex::BitVector<T>& bv) const {
    return hash<std::string>()(bv.to_string());
  }
};

}