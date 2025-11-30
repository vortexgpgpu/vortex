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

#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>

#include "svdpi.h"
#include "verilated_vpi.h"

#ifdef XLEN_64
#define iword_t   int64_t
#define uword_t   uint64_t
#define idword_t  __int128_t
#define udword_t  __uint128_t
#else
#define iword_t   int32_t
#define uword_t   uint32_t
#define idword_t  int64_t
#define udword_t  uint64_t
#endif

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 3
#endif

extern "C" {
  void dpi_imul(bool enable, bool is_signed_a, bool is_signed_b, iword_t a, iword_t b, iword_t* resultl, iword_t* resulth);
  void dpi_idiv(bool enable, bool is_signed, iword_t a, iword_t b, iword_t* quotient, iword_t* remainder);

  int dpi_register();
  void dpi_assert(int inst, bool cond, int delay);

  void dpi_trace(int level, const char* format, ...);
  void dpi_trace_start();
  void dpi_trace_stop();
}

bool sim_trace_enabled();
void sim_trace_enable(bool enable);

class ShiftRegister {
public:
  ShiftRegister() : init_(false), depth_(0) {}

  void ensure_init(int depth) {
    if (!init_) {
      buffer_.resize(depth);
      init_  = true;
      depth_ = depth;
    }
  }

  void push(int value, bool enable) {
    if (!enable)
      return;
    for (unsigned i = 0; i < depth_-1; ++i) {
      buffer_[i] = buffer_[i+1];
    }
    buffer_[depth_-1] = value;
  }

  int top() const {
    return buffer_[0];
  }

private:

  std::vector<int> buffer_;
  bool init_;
  unsigned depth_;
};

class Instances {
public:
  ShiftRegister& get(int inst) {
    return instances_.at(inst);
  }

  int allocate() {
    mutex_.lock();
    int inst = instances_.size();
    instances_.resize(inst + 1);
    mutex_.unlock();
    return inst;
  }

private:
  std::vector<ShiftRegister> instances_;
  std::mutex mutex_;
};

Instances instances;

int dpi_register() {
  return instances.allocate();
}

void dpi_assert(int inst, bool cond, int delay) {
  ShiftRegister& sr = instances.get(inst);

  sr.ensure_init(delay);
  sr.push(!cond, 1);

  auto status = sr.top();
  if (status) {
    printf("delayed assertion at %s!\n", svGetNameFromScope(svGetScope()));
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

void dpi_imul(bool enable, bool is_signed_a, bool is_signed_b, iword_t a, iword_t b, iword_t* resultl, iword_t* resulth) {
  if (!enable)
    return;
  udword_t first  = *(uword_t*)&a;
  udword_t second = *(uword_t*)&b;

  udword_t mask = udword_t(-1) << (8 * sizeof(iword_t));

  if (is_signed_a && a < 0) {
    first |= mask;
  }

  if (is_signed_b && b < 0) {
    second |= mask;
  }

  udword_t result;
  if (is_signed_a || is_signed_b) {
    result = idword_t(first) * idword_t(second);
  } else {
    result = first * second;
  }

  *resultl = iword_t(result);
  *resulth = iword_t(result >> (8 * sizeof(iword_t)));
}

void dpi_idiv(bool enable, bool is_signed, iword_t a, iword_t b, iword_t* quotient, iword_t* remainder) {
  if (!enable)
    return;

  uword_t dividen = a;
  uword_t divisor = b;

  auto inf_neg = uword_t(1) << (8 * sizeof(iword_t) - 1);

  if (is_signed) {
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else if (dividen == inf_neg && divisor == -1) {
      *remainder = 0;
      *quotient  = dividen;
    } else {
      *quotient  = (iword_t)dividen / (iword_t)divisor;
      *remainder = (iword_t)dividen % (iword_t)divisor;
    }
  } else {
    if (b == 0) {
      *quotient  = -1;
      *remainder = dividen;
    } else {
      *quotient  = dividen / divisor;
      *remainder = dividen % divisor;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

void dpi_trace(int level, const char* format, ...) {
  if (level > DEBUG_LEVEL)
    return;
  if (!sim_trace_enabled())
    return;
  va_list va;
	va_start(va, format);
	vprintf(format, va);
	va_end(va);
}

void dpi_trace_start() {
  sim_trace_enable(true);
}

void dpi_trace_stop() {
  sim_trace_enable(false);
}
