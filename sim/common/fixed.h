#pragma once

#include <cstdint>
#include <cstdlib>
#include <assert.h>

template <uint32_t F, typename T = int32_t>
class Fixed {
private:

  template <uint32_t F2, typename T2> 
  struct Cast {
  private:
    template <bool isF2Bigger, bool isT2Bigger> struct Tag {};

    inline static T Convert(T2 value, Tag<false, false>) {
      return static_cast<T>(value) << (F - F2);
    }

    inline static T Convert(T2 value, Tag<false, true>) {
      return static_cast<T>(value) >> (F2 - F);
    }

    inline static T Convert(T2 value, Tag<true, false>) {
      return static_cast<T>(value << (F - F2));
    }

    inline static T Convert(T2 value, Tag<true, true>) {
      return static_cast<T>(value >> (F2 - F));
    }

  public:    
    inline static T Convert(T2 value) {
      return Convert(value, Tag<(sizeof(T2) > sizeof(T)), (F2 > F)>{});
    }  
  };

public:
  using data_type = T;

  static constexpr uint32_t FRAC = F;
  static constexpr uint32_t INT = sizeof(T) * 8 - FRAC;
  static constexpr uint32_t HFRAC = FRAC >> 1;
  static constexpr T ONE = static_cast<T>(1) << FRAC;
  static constexpr T MASK = ONE - 1;
  static constexpr T IMASK = ~MASK;
  static constexpr T HALF = ONE >> 1;
  static constexpr T TWO = ONE << 1;

  Fixed() {}

  explicit Fixed(int64_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(uint64_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(int32_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(uint32_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(int16_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(uint16_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(int8_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  explicit Fixed(uint8_t rhs)
      : data_(static_cast<T>(rhs << FRAC)) {
    assert((static_cast<int64_t>(rhs) << FRAC) == data_);
  }

  template <uint32_t F2, typename T2>
  explicit Fixed(Fixed<F2, T2> rhs)
    : data_(Cast<F2, T2>::Convert(rhs.data()))
  {}

  explicit Fixed(float rhs)
      : data_(static_cast<T>(rhs * ONE)) {
    assert(data_ == static_cast<T>(rhs * ONE));
  }

  bool operator==(Fixed rhs) const {
    return (data_ == rhs.data_);
  }

  bool operator!=(Fixed rhs) const {
    return (data_ != rhs.data_);
  }

  bool operator<(Fixed rhs) const {
    return (data_ < rhs.data_);
  }

  bool operator<=(Fixed rhs) const {
    return (data_ <= rhs.data_);
  }

  bool operator>(Fixed rhs) const {
    return (data_ > rhs.data_);
  }

  bool operator>=(Fixed rhs) const {
    return (data_ >= rhs.data_);
  }

  Fixed operator-() const {
    return make(-data_);
  }

  Fixed operator+=(Fixed rhs) {
    *this = (*this) + rhs;
    return *this;
  }

  Fixed operator-=(Fixed rhs) {
    *this = (*this) - rhs;
    return *this;
  }

  Fixed operator*=(Fixed rhs) {
    *this = (*this) * rhs;
    return *this;
  }

  Fixed operator/=(Fixed rhs) {
    *this = (*this) / rhs;
    return *this;
  }

  template <uint32_t F2, typename T2>
  Fixed operator*=(Fixed<F2, T2> rhs) {
    *this = (*this) * rhs;
    return *this;
  }

  template <uint32_t F2, typename T2>
  Fixed operator/=(Fixed<F2, T2> rhs) {
    *this = (*this) / rhs;
    return *this;
  }

  Fixed operator*=(int32_t rhs) {
    *this = (*this) * rhs;
    return *this;
  }

  Fixed operator*=(uint32_t rhs) {
    *this = (*this) * rhs;
    return *this;
  }

  Fixed operator*=(float rhs) {
    *this = (*this) * rhs;
    return *this;
  }

  Fixed operator/=(int32_t rhs) {
    *this = (*this) / rhs;
    return *this;
  }

  Fixed operator/=(uint32_t rhs) {
    *this = (*this) / rhs;
    return *this;
  }

  Fixed operator/=(float rhs) {
    *this = (*this) / rhs;
    return *this;
  }

  friend Fixed operator+(Fixed lhs, Fixed rhs) {
    assert((static_cast<int64_t>(lhs.data_) + rhs.data_) ==
           (lhs.data_ + rhs.data_));
    return Fixed::make(lhs.data_ + rhs.data_);
  }

  friend Fixed operator-(Fixed lhs, Fixed rhs) {
    assert((static_cast<int64_t>(lhs.data_) - rhs.data_) ==
           (lhs.data_ - rhs.data_));
    return Fixed::make(lhs.data_ - rhs.data_);
  }

  friend Fixed operator*(Fixed lhs, Fixed rhs) {
    return Fixed::make((static_cast<int64_t>(lhs.data_) * rhs.data_) >> FRAC);
  }

  template <uint32_t F2, typename T2>
  friend Fixed operator*(Fixed lhs, Fixed<F2, T2> rhs) {
    return Fixed::make((static_cast<int64_t>(lhs.data_) * rhs.data()) >> F2);
  }

  friend Fixed operator/(Fixed lhs, Fixed rhs) {
    assert(rhs.data_ != 0);
    return Fixed::make((static_cast<int64_t>(lhs.data_) << FRAC) / rhs.data_);
  }

  template <uint32_t F2, typename T2>
  friend Fixed operator/(Fixed lhs, Fixed<F2, T2> rhs) {
    assert(rhs.data() != 0);
    return Fixed::make((static_cast<int64_t>(lhs.data_) << F2) / rhs.data());
  }

  friend Fixed operator*(Fixed lhs, float rhs) {
    return static_cast<float>(lhs) * rhs;
  }

  friend Fixed operator*(float lhs, Fixed rhs) {
    return lhs * static_cast<float>(rhs);
  }

  friend Fixed operator/(Fixed lhs, float rhs) {
    return static_cast<float>(lhs) / rhs;
  }

  friend Fixed operator/(float lhs, Fixed rhs) {
    return lhs / static_cast<float>(rhs);
  }

  friend Fixed operator*(Fixed lhs, char rhs) {
    return lhs * static_cast<int32_t>(rhs);
  }

  friend Fixed operator*(char lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, char rhs) {
    return lhs / static_cast<int32_t>(rhs);
  }

  friend Fixed operator/(char lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator*(Fixed lhs, uint8_t rhs) {
    return lhs * static_cast<int32_t>(rhs);
  }

  friend Fixed operator*(uint8_t lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, uint8_t rhs) {
    return lhs / static_cast<int32_t>(rhs);
  }

  friend Fixed operator/(uint8_t lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator*(Fixed lhs, short rhs) {
    return lhs * static_cast<int32_t>(rhs);
  }

  friend Fixed operator*(short lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, short rhs) {
    return lhs / static_cast<int32_t>(rhs);
  }

  friend Fixed operator/(short lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator*(Fixed lhs, uint16_t rhs) {
    return lhs * static_cast<int32_t>(rhs);
  }

  friend Fixed operator*(uint16_t lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, uint16_t rhs) {
    return lhs / static_cast<int32_t>(rhs);
  }

  friend Fixed operator/(uint16_t lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator*(Fixed lhs, int32_t rhs) {
    auto value = static_cast<T>(lhs.data_ * rhs);
    assert((lhs.data_ * static_cast<int64_t>(rhs)) == value);
    return Fixed::make(value);
  }

  friend Fixed operator*(int32_t lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, int32_t rhs) {
    assert(rhs);
    auto value = static_cast<T>(lhs.data_ / rhs);
    return Fixed::make(value);
  }

  friend Fixed operator/(int32_t lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator*(Fixed lhs, uint32_t rhs) {
    auto value = static_cast<T>(lhs.data_ << rhs);
    assert((lhs.data_ << static_cast<int64_t>(rhs)) == value);
    return Fixed::make(value);
  }

  friend Fixed operator*(uint32_t lhs, Fixed rhs) {
    return rhs * lhs;
  }

  friend Fixed operator/(Fixed lhs, uint32_t rhs) {
    assert(rhs);
    auto value = static_cast<T>(lhs.data_ / rhs);
    return Fixed::make(value);
  }

  friend Fixed operator/(uint32_t lhs, Fixed rhs) {
    return rhs / lhs;
  }

  friend Fixed operator<<(Fixed lhs, int32_t rhs) {
    auto value = static_cast<T>(lhs.data_ << rhs);
    assert((lhs.data_ << static_cast<int64_t>(rhs)) == value);
    return Fixed::make(value);
  }

  friend Fixed operator>>(Fixed lhs, int32_t rhs) {
    auto value = static_cast<T>(lhs.data_ >> rhs);
    return Fixed::make(value);
  }

  friend Fixed operator<<(Fixed lhs, uint32_t rhs) {
    auto value = static_cast<T>(lhs.data_ << rhs);
    assert((lhs.data_ << static_cast<int64_t>(rhs)) == value);
    return Fixed::make(value);
  }

  friend Fixed operator>>(Fixed lhs, uint32_t rhs) {
    auto value = static_cast<T>(lhs.data_ >> rhs);
    return Fixed::make(value);
  }

  static Fixed make(T value) {
    Fixed ret;
    ret.data_ = value;
    return ret;
  }

  explicit operator int64_t() const {
    return static_cast<int64_t>(data_ >> F);
  }

  explicit operator uint64_t() const {
    return static_cast<uint64_t>(data_ >> F);
  }

  explicit operator int32_t() const {
    return static_cast<int32_t>(data_ >> F);
  }

  explicit operator uint32_t() const {
    return static_cast<uint32_t>(data_ >> F);
  }

  explicit operator int16_t() const {
    return static_cast<int16_t>(data_ >> F);
  }

  explicit operator uint16_t() const {
    return static_cast<uint16_t>(data_ >> F);
  }

  explicit operator int8_t() const {
    return static_cast<int8_t>(data_ >> F);
  }

  explicit operator uint8_t() const {
    return static_cast<uint8_t>(data_ >> F);
  }

  template <uint32_t F2, typename T2>
  explicit operator Fixed<F2, T2>() const {
    return Fixed<F2, T2>(*this);
  }

  explicit operator float() const {
    return static_cast<float>(data_) / (static_cast<T>(1) << F);
  }

  T data() const {
    return data_;
  }

private:
  T data_;
};