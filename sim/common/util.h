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

#include <algorithm>
#include <array>
#include <assert.h>
#include <bitmanip.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

namespace vortex {

template <typename... Args>
void unused(Args &&...) {}

#define __unused(...) unused(__VA_ARGS__)

#define __assert(cond, msg)                           \
  if (!(cond)) {                                      \
    std::cerr << "Assertion failed: " << msg << "\n"; \
    std::cerr << "File: " << __FILE__ << "\n";        \
    std::cerr << "Line: " << __LINE__ << "\n";        \
    std::cerr << "Function: " << __func__ << "\n";    \
    std::abort();                                     \
  }

// return file extension
const char *fileExtension(const char *filepath);

#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING_UNUSED_PARAMETER \
  __pragma(warning(disable : 4100))
#define DISABLE_WARNING_UNREFERENCED_FUNCTION __pragma(warning(disable : 4505))
#define DISABLE_WARNING_ANONYMOUS_STRUCT __pragma(warning(disable : 4201))
#define DISABLE_WARNING_UNUSED_VARIABLE __pragma(warning(disable : 4189))
#define DISABLE_WARNING_MISSING_FIELD_INITIALIZERS __pragma(warning(disable : 4351))
#elif defined(__GNUC__)
#define DISABLE_WARNING_PUSH _Pragma("GCC diagnostic push")
#define DISABLE_WARNING_POP _Pragma("GCC diagnostic pop")
#define DISABLE_WARNING_UNUSED_PARAMETER \
  _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION \
  _Pragma("GCC diagnostic ignored \"-Wunused-function\"")
#define DISABLE_WARNING_ANONYMOUS_STRUCT \
  _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#define DISABLE_WARNING_UNUSED_VARIABLE \
  _Pragma("GCC diagnostic ignored \"-Wunused-but-set-variable\"")
#define DISABLE_WARNING_MISSING_FIELD_INITIALIZERS \
  _Pragma("GCC diagnostic ignored \"-Wmissing-field-initializers\"")
#define DISABLE_WARNING_STRICT_ALIASING \
  _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"")
#elif defined(__clang__)
#define DISABLE_WARNING_PUSH _Pragma("clang diagnostic push")
#define DISABLE_WARNING_POP _Pragma("clang diagnostic pop")
#define DISABLE_WARNING_UNUSED_PARAMETER \
  _Pragma("clang diagnostic ignored \"-Wunused-parameter\"")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION \
  _Pragma("clang diagnostic ignored \"-Wunused-function\"")
#define DISABLE_WARNING_ANONYMOUS_STRUCT \
  _Pragma("clang diagnostic ignored \"-Wgnu-anonymous-struct\"")
#define DISABLE_WARNING_UNUSED_VARIABLE \
  _Pragma("clang diagnostic ignored \"-Wunused-but-set-variable\"")
#define DISABLE_WARNING_MISSING_FIELD_INITIALIZERS \
  _Pragma("clang diagnostic ignored \"-Wmissing-field-initializers\"")
#define DISABLE_WARNING_STRICT_ALIASING \
  _Pragma("clang diagnostic ignored \"-Wstrict-aliasing\"")
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNUSED_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
#define DISABLE_WARNING_ANONYMOUS_STRUCT
#define DISABLE_WARNING_STRICT_ALIASING
#endif

void *aligned_malloc(size_t size, size_t alignment);
void aligned_free(void *ptr);

// Verilator data type casting
template <typename R, size_t W, typename Enable = void>
class VDataCast;
template <typename R, size_t W>
class VDataCast<R, W, typename std::enable_if<(W > 8)>::type> {
public:
  template <typename T>
  static R get(T &obj) {
    return reinterpret_cast<R>(obj.data());
  }
};
template <typename R, size_t W>
class VDataCast<R, W, typename std::enable_if<(W <= 8)>::type> {
public:
  template <typename T>
  static R get(T &obj) {
    return reinterpret_cast<R>(&obj);
  }
};

template <typename T, std::size_t N, typename... Args, std::size_t... Is>
constexpr std::array<T, N> make_array_impl(std::index_sequence<Is...>, Args &&...args) {
  return {{(static_cast<void>(Is), T(std::forward<Args>(args)...))...}};
}

template <typename T, std::size_t N, typename... Args>
constexpr std::array<T, N> make_array(Args &&...args) {
  return make_array_impl<T, N>(std::make_index_sequence<N>{}, std::forward<Args>(args)...);
}

// visit_var(variant, f1, f2, f3, ...)
//   - deduces a closure type that inherits all your lambdas
//   - forwards them into std::visit
//   - works in C++17 without any extra global templates
template <typename Variant, typename... Fs>
auto visit_var(Variant &&var, Fs &&...fs) {
  // define a local visitor type that inherits all your lambdas
  struct Visitor : std::decay_t<Fs>... {
    // inherit ctors
    Visitor(Fs &&...f) : std::decay_t<Fs>(std::forward<Fs>(f))... {}
    // pull in operator() into this scope
    using std::decay_t<Fs>::operator()...;
  };

  return std::visit(
      Visitor{std::forward<Fs>(fs)...},
      std::forward<Variant>(var));
}

template <typename To, typename From>
To bit_cast(const From& src) {
  union cast_t { From from; To to; };
  cast_t cast{0};
  cast.from = src;
  return cast.to;
}

std::string to_hex_str(uint32_t v);

std::string resolve_file_path(const std::string &filename, const std::string &searchPaths);

} // namespace vortex