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

#include <array>
#include <cstdint>
#include "verilated.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>	// Trace file format header
#endif

template <typename T>
class vl_simulator {
private:

  T top_;
#ifdef VCD_OUTPUT
  VerilatedVcdC tfp_;
#endif

public:

  vl_simulator() {
    top_.clk = 0;
    top_.reset = 0;
  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    top_.trace(&tfp_, 99);
    tfp_.open("trace.vcd");
  #endif
  }

  ~vl_simulator() {
  #ifdef VCD_OUTPUT
    tfp_.close();
  #endif
    top_.final();
  }

  uint64_t reset(uint64_t ticks) {
    top_.reset = 1;
    ticks = this->step(ticks, 2);
    top_.reset = 0;
    return ticks;
  }

  uint64_t step(uint64_t ticks, uint32_t count = 1) {
    while (count--) {
      top_.eval();
    #ifdef VCD_OUTPUT
      tfp_.dump(ticks);
    #endif
      top_.clk = !top_.clk;
      ++ticks;
    }
    return ticks;
  }

  T* operator->() {
    return &top_;
  }
};

template <typename... Args>
void vl_setw(uint32_t* sig, Args&&... args) {
  std::array<uint32_t, sizeof... (Args)> arr{static_cast<uint32_t>(std::forward<Args>(args))...};
  for (size_t i = 0; i < sizeof... (Args); ++i) {
    sig[i] = arr[i];
  }
}

template <typename... Args>
int  vl_cmpw(const uint32_t* sig, Args&&... args) {
  std::array<uint32_t, sizeof... (Args)> arr{static_cast<uint32_t>(std::forward<Args>(args))...};
  for (size_t i = 0; i < sizeof... (Args); ++i) {
    if (sig[i] < arr[i])
      return -1;
    if (sig[i] > arr[i])
      return 1;
  }
  return 0;
}