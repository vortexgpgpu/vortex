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

#include <cassert>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include "util.h"

namespace vortex {

///////////////////////////////////////////////////////////////////////////////
// SmallFunction: a tiny std::function-like type erasure with inline storage.
//
// Goals:
// - No heap allocation for small lambdas (common case: capture a pointer)
// - Still supports larger callables via heap fallback
// - Copyable (needed for std::vector fill construction of SimChannel)
///////////////////////////////////////////////////////////////////////////////

template <typename Sig, size_t InlineBytes>
class SmallFunction;

template <typename R, typename... Args, size_t InlineBytes>
class SmallFunction<R(Args...), InlineBytes> {
public:
  SmallFunction() : vtbl_(nullptr), heap_(false), heap_ptr_(nullptr) {}

  SmallFunction(std::nullptr_t) : SmallFunction() {}

  SmallFunction(const SmallFunction& other) : SmallFunction() {
    if (other.vtbl_) {
      other.vtbl_->copy(this, &other);
    }
  }

  SmallFunction(SmallFunction&& other) noexcept : SmallFunction() {
    if (other.vtbl_) {
      other.vtbl_->move(this, &other);
    }
  }

  template <typename F>
  SmallFunction(F f) {
    this->assign(std::move(f));
  }

  ~SmallFunction() {
    this->reset();
  }

  SmallFunction& operator=(const SmallFunction& other) {
    if (this != &other) {
      this->reset();
      if (other.vtbl_) {
        other.vtbl_->copy(this, &other);
      }
    }
    return *this;
  }

  SmallFunction& operator=(SmallFunction&& other) noexcept {
    if (this != &other) {
      this->reset();
      if (other.vtbl_) {
        other.vtbl_->move(this, &other);
      }
    }
    return *this;
  }

  template <typename F>
  SmallFunction& operator=(F f) {
    this->reset();
    this->assign(std::move(f));
    return *this;
  }

  explicit operator bool() const { return vtbl_ != nullptr; }

  R operator()(Args... args) const {
    __assert(vtbl_ != nullptr, "SmallFunction: call on empty");
    return vtbl_->invoke(this, std::forward<Args>(args)...);
  }

  void reset() {
    if (vtbl_) {
      vtbl_->destroy(this);
      vtbl_ = nullptr;
      heap_ = false;
      heap_ptr_ = nullptr;
    }
  }

private:
  struct VTable {
    R (*invoke)(const SmallFunction*, Args&&...);
    void (*destroy)(SmallFunction*);
    void (*copy)(SmallFunction* dst, const SmallFunction* src);
    void (*move)(SmallFunction* dst, SmallFunction* src);
  };

  template <typename F>
  static R invoke_impl(const SmallFunction* self, Args&&... args) {
    if (self->heap_) {
      return (*static_cast<const F*>(self->heap_ptr_))(std::forward<Args>(args)...);
    }
    return (*reinterpret_cast<const F*>(&self->inline_storage_))(std::forward<Args>(args)...);
  }

  template <typename F>
  static void destroy_impl(SmallFunction* self) {
    if (self->heap_) {
      delete static_cast<F*>(self->heap_ptr_);
      self->heap_ptr_ = nullptr;
      self->heap_ = false;
    } else {
      auto* obj = reinterpret_cast<F*>(&self->inline_storage_);
      obj->~F();
    }
  }

  template <typename F>
  static void copy_impl(SmallFunction* dst, const SmallFunction* src) {
    dst->vtbl_ = src->vtbl_;
    if (src->heap_) {
      dst->heap_ptr_ = new F(*static_cast<const F*>(src->heap_ptr_));
      dst->heap_ = true;
    } else {
      new (&dst->inline_storage_) F(*reinterpret_cast<const F*>(&src->inline_storage_));
      dst->heap_ = false;
      dst->heap_ptr_ = nullptr;
    }
  }

  template <typename F>
  static void move_impl(SmallFunction* dst, SmallFunction* src) {
    dst->vtbl_ = src->vtbl_;
    if (src->heap_) {
      dst->heap_ptr_ = src->heap_ptr_;
      dst->heap_ = true;
      src->heap_ptr_ = nullptr;
      src->heap_ = false;
      src->vtbl_ = nullptr;
    } else {
      new (&dst->inline_storage_) F(std::move(*reinterpret_cast<F*>(&src->inline_storage_)));
      dst->heap_ = false;
      dst->heap_ptr_ = nullptr;
      reinterpret_cast<F*>(&src->inline_storage_)->~F();
      src->vtbl_ = nullptr;
    }
  }

  template <typename F>
  void assign(F&& f) {
    using Fn = std::decay_t<F>;
    static_assert(std::is_invocable_r_v<R, Fn, Args...>, "SmallFunction: signature mismatch");
    static const VTable vtbl = {
      &invoke_impl<Fn>,
      &destroy_impl<Fn>,
      &copy_impl<Fn>,
      &move_impl<Fn>
    };
    vtbl_ = &vtbl;

    if constexpr (sizeof(Fn) <= InlineBytes && alignof(Fn) <= alignof(std::max_align_t)) {
      heap_ = false;
      heap_ptr_ = nullptr;
      new (&inline_storage_) Fn(std::forward<F>(f));
    } else {
      heap_ = true;
      heap_ptr_ = new Fn(std::forward<F>(f));
    }
  }

  const VTable* vtbl_;
  bool heap_;
  void* heap_ptr_;
  std::aligned_storage_t<InlineBytes, alignof(std::max_align_t)> inline_storage_;
};

} // namespace vortex
