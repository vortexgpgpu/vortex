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

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include <bitmanip.h>

template <typename... Args>
void unused(Args&&...) {}

#define __unused(...) unused(__VA_ARGS__)

// return file extension
const char* fileExtension(const char* filepath);

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
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNUSED_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
#define DISABLE_WARNING_ANONYMOUS_STRUCT
#endif

void *aligned_malloc(size_t size, size_t alignment);
void aligned_free(void *ptr);

namespace vortex {

// Verilator data type casting
template <typename R, size_t W, typename Enable = void>
class VDataCast;
template <typename R, size_t W>
class VDataCast<R, W, typename std::enable_if<(W > 8)>::type> {
public:
  template <typename T>
  static R get(T& obj) {
    return reinterpret_cast<R>(obj.data());
  }
};
template <typename R, size_t W>
class VDataCast<R, W, typename std::enable_if<(W <= 8)>::type> {
public:
  template <typename T>
  static R get(T& obj) {
    return reinterpret_cast<R>(&obj);
  }
};

}