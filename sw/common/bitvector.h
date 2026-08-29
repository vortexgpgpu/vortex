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

  class BitProxy {
  public:
    BitProxy(BitVector& bv, size_t pos) : bv_(bv), pos_(pos) {}

    operator bool() const {
      return bv_.test(pos_);
    }

    BitProxy& operator=(bool value) {
      bv_.set(pos_, value);
      return *this;
    }

    BitProxy& operator=(const BitProxy& other) {
      bool value = other.bv_.test(other.pos_);
      bv_.set(pos_, value);
      return *this;
    }

  private:
    BitVector& bv_;
    size_t pos_;
  };

  static constexpr size_t BITS_PER_WORD = sizeof(T) * 8;
  size_t size_;
  T single_word_;
  std::vector<T> words_;
  bool all_zero_;

  constexpr size_t wordIndex(size_t pos) const {
    return pos / BITS_PER_WORD;
  }

  constexpr T wordOffset(size_t pos) const {
    return T(1) << (pos % BITS_PER_WORD);
  }

  constexpr T bitMask(size_t size) const {
    if (size == 0)
      return 0;
    if (size < BITS_PER_WORD) {
      return (T(1) << size) - 1;
    }
    return ~T(0);
  }

  void updateAllZero() {
    if (size_ <= BITS_PER_WORD) {
      all_zero_ = (single_word_ == 0);
    } else {
      all_zero_ = std::all_of(words_.begin(), words_.end(), [](T word) { return word == 0; });
    }
  }

  void clearUnusedBits() {
    if (size_ <= BITS_PER_WORD) {
      single_word_ &= bitMask(size_);
    } else {
      size_t last_word_bits = size_ % BITS_PER_WORD;
      if (last_word_bits != 0) {
        words_.back() &= bitMask(last_word_bits);
      }
    }
  }

public:

  explicit BitVector(size_t size = 0)
    : size_(size)
    , all_zero_(true) {
    if (size <= BITS_PER_WORD) {
      single_word_ = 0;
    } else {
      size_t num_blocks = (size + (BITS_PER_WORD - 1)) / BITS_PER_WORD;
      words_.resize(num_blocks, 0);
    }
  }

  BitVector(size_t size, T value)
    : size_(size) {
    if (size_ <= BITS_PER_WORD) {
      single_word_ = value;
    } else {
      size_t num_blocks = (size_ + BITS_PER_WORD - 1) / BITS_PER_WORD;
      words_.resize(num_blocks, 0);
      words_[0] = value;
    }
    this->clearUnusedBits();
    this->updateAllZero();
  }

  BitVector(const BitVector& other)
    : size_(other.size_)
    , single_word_(other.single_word_)
    , words_(other.words_)
    , all_zero_(other.all_zero_)
  {}

  BitVector(BitVector&& other) noexcept
    : size_(other.size_)
    , single_word_(other.single_word_)
    , words_(std::move(other.words_))
    , all_zero_(other.all_zero_) {
    other.size_ = 0;
    other.single_word_ = 0;
    other.all_zero_ = true;
  }

  ~BitVector() {}

  void set(size_t pos) {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    if (size_ <= BITS_PER_WORD) {
      single_word_ |= this->wordOffset(pos);
    } else {
      words_[this->wordIndex(pos)] |= this->wordOffset(pos);
    }
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
    if (size_ <= BITS_PER_WORD) {
      single_word_ = 0;
    } else {
      std::fill(words_.begin(), words_.end(), 0);
    }
    all_zero_ = true;
  }

  void reset(size_t pos) {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    if (size_ <= BITS_PER_WORD) {
      single_word_ &= ~this->wordOffset(pos);
    } else {
      words_[this->wordIndex(pos)] &= ~this->wordOffset(pos);
    }
    this->updateAllZero();
  }

  bool test(size_t pos) const {
    if (pos >= size_) throw std::out_of_range("Index out of range");
    if (size_ <= BITS_PER_WORD) {
      return single_word_ & this->wordOffset(pos);
    } else {
      return words_[this->wordIndex(pos)] & this->wordOffset(pos);
    }
  }

  size_t size() const {
    return size_;
  }

  void resize(size_t new_size) {
    if (new_size == size_)
      return;
    if (new_size <= BITS_PER_WORD) {
      T new_word;
      if (size_ <= BITS_PER_WORD) {
        new_word = single_word_;
      } else {
        new_word = words_.at(0);
        words_.clear();
      }
      single_word_ = new_word;
    } else {
      size_t num_blocks = (new_size + (BITS_PER_WORD - 1)) / BITS_PER_WORD;
      words_.resize(num_blocks, 0);
    }
    size_ = new_size;
    this->clearUnusedBits();
    this->updateAllZero();
  }

  BitVector& operator=(const BitVector& other) {
    if (this != &other) {
      size_ = other.size_;
      single_word_ = other.single_word_;
      words_ = other.words_;
      all_zero_ = other.all_zero_;
    }
    return *this;
  }

  BitVector& operator=(BitVector&& other) noexcept {
    if (this != &other) {
      size_ = other.size_;
      single_word_ = other.single_word_;
      words_ = std::move(other.words_);
      all_zero_ = other.all_zero_;
      other.size_ = 0;
      other.single_word_ = 0;
      other.all_zero_ = true;
    }
    return *this;
  }

  bool operator==(const BitVector& other) const {
    if (size_ != other.size_)
      return false;
    if (size_ <= BITS_PER_WORD) {
      return single_word_ == other.single_word_;
    } else {
      return (words_ == other.words_);
    }
  }

  bool operator!=(const BitVector& other) const {
    return !(*this == other);
  }

  bool operator[](size_t pos) const {
    return test(pos);
  }

  BitProxy operator[](size_t pos) {
    return BitProxy(*this, pos);
  }

  BitVector& operator&=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    if (size_ <= BITS_PER_WORD) {
      single_word_ &= other.single_word_;
    } else {
      for (size_t i = 0; i < words_.size(); ++i) {
        words_[i] &= other.words_[i];
      }
    }
    this->updateAllZero();
    return *this;
  }

  BitVector& operator|=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    if (size_ <= BITS_PER_WORD) {
      single_word_ |= other.single_word_;
    } else {
      for (size_t i = 0; i < words_.size(); ++i) {
        words_[i] |= other.words_[i];
      }
    }
    this->updateAllZero();
    return *this;
  }

  BitVector& operator^=(const BitVector& other) {
    if (size_ != other.size_) throw std::invalid_argument("Bit sizes must match");
    if (size_ <= BITS_PER_WORD) {
      single_word_ ^= other.single_word_;
    } else {
      for (size_t i = 0; i < words_.size(); ++i) {
        words_[i] ^= other.words_[i];
      }
    }
    this->updateAllZero();
    return *this;
  }

  void flip() {
    if (size_ <= BITS_PER_WORD) {
      single_word_ = ~single_word_;
    } else {
      for (auto &word : words_) {
        word = ~word;
      }
    }
    this->clearUnusedBits();
    this->updateAllZero();
  }

  BitVector operator~() const {
    BitVector result(*this);
    result.flip();
    return result;
  }

  void reverse() {
    if (size_ == 0)
      return;
    if (size_ <= BITS_PER_WORD) {
      single_word_ = static_cast<T>(bit_reverse(single_word_, BITS_PER_WORD));
    } else {
      size_t remaining_bits = size_ % BITS_PER_WORD;
      if (remaining_bits != 0) {
        std::vector<T> reversed_words(words_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
          size_t reversed_pos = size_ - 1 - i;
          size_t src_word = i / BITS_PER_WORD;
          size_t src_offset = i % BITS_PER_WORD;
          size_t dst_word = reversed_pos / BITS_PER_WORD;
          size_t dst_offset = reversed_pos % BITS_PER_WORD;
          if (words_[src_word] & (T(1) << src_offset)) {
            reversed_words[dst_word] |= (T(1) << dst_offset);
          }
        }
        words_ = std::move(reversed_words);
      } else {
        std::reverse(words_.begin(), words_.end());
        for (auto &word : words_) {
          word = static_cast<T>(bit_reverse(word, BITS_PER_WORD));
        }
      }
    }
  }

  size_t count() const {
   if (size_ <= BITS_PER_WORD) {
      if constexpr (sizeof(T) <= sizeof(unsigned int)) {
        return __builtin_popcount(single_word_);
      } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return __builtin_popcountl(single_word_);
      } else {
        return __builtin_popcountll(single_word_);
      }
    } else {
      size_t count = 0;
      for (auto word : words_) {
        if constexpr (sizeof(T) <= sizeof(unsigned int)) {
          count += __builtin_popcount(word);
        } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
          count += __builtin_popcountl(word);
        } else {
          count += __builtin_popcountll(word);
        }
      }
      return count;
    }
  }

  bool none() const {
    return all_zero_;
  }

  bool any() const {
    return !all_zero_;
  }

  bool all() const {
    size_t remaining_bits = size_ % BITS_PER_WORD;
    T full_mask = ~T(0);
    T rem_mask = (T(1) << remaining_bits) - 1;
    if (size_ <= BITS_PER_WORD) {
      auto expected = (remaining_bits != 0) ? rem_mask : full_mask;
      return (single_word_ == expected);
    } else {
      size_t num_blocks = size_ / BITS_PER_WORD;
      for (size_t i = 0; i < num_blocks; ++i) {
        if (words_[i] != full_mask)
          return false;
      }
      if (remaining_bits != 0) {
        if ((words_[num_blocks] & rem_mask) != rem_mask)
          return false;
      }
      return true;
    }
  }

   BitVector& operator<<=(size_t shift) {
    if (shift >= size_) {
      reset();
      return *this;
    }
    if (size_ <= BITS_PER_WORD) {
      single_word_ = single_word_ << shift;
    } else {
      size_t word_shift = shift / BITS_PER_WORD;
      size_t bit_shift = shift % BITS_PER_WORD;

      if (word_shift > 0) {
        for (size_t i = words_.size() - 1; i >= word_shift; --i) {
          words_[i] = words_[i - word_shift];
        }
        std::fill(words_.begin(), words_.begin() + word_shift, 0);
      }

      if (bit_shift > 0) {
        for (size_t i = words_.size() - 1; i > 0; --i) {
          words_[i] = (words_[i] << bit_shift) | (words_[i - 1] >> (BITS_PER_WORD - bit_shift));
        }
        words_[0] <<= bit_shift;
      }
    }
    this->clearUnusedBits();
    this->updateAllZero();
    return *this;
  }

  BitVector& operator>>=(size_t shift) {
    if (shift >= size_) {
      reset();
      return *this;
    }
    if (size_ <= BITS_PER_WORD) {
      single_word_ >>= shift;
    } else {
      size_t word_shift = shift / BITS_PER_WORD;
      size_t bit_shift = shift % BITS_PER_WORD;

      if (word_shift > 0) {
        for (size_t i = 0; i < words_.size() - word_shift; ++i) {
          words_[i] = words_[i + word_shift];
        }
        std::fill(words_.end() - word_shift, words_.end(), 0);
      }

      if (bit_shift > 0) {
        for (size_t i = 0; i < words_.size() - 1; ++i) {
          words_[i] = (words_[i] >> bit_shift) | (words_[i + 1] << (BITS_PER_WORD - bit_shift));
        }
        words_.back() >>= bit_shift;
      }
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

  friend BitVector operator<<(const BitVector& lhs, size_t shift) {
    BitVector result(lhs);
    result <<= shift;
    return result;
  }

  friend BitVector operator>>(const BitVector& lhs, size_t shift) {
    BitVector result(lhs);
    result >>= shift;
    return result;
  }
};

}

namespace std {

template <typename T>
struct hash<vortex::BitVector<T>> {
  size_t operator()(const vortex::BitVector<T>& bv) const {
    size_t seed = bv.size_;
    for (auto word : bv.words_) {
      seed ^= hash<T>()(word) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}