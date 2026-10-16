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

#include <cstdint>

// Bit-exact model of the MUFU datapath (hw/rtl/mufu).
namespace vortex::mufu {

enum fn_t : uint32_t {
  FN_RCP   = 0,
  FN_RSQRT = 1,
  FN_EX2   = 2,
  FN_LG2   = 3,
  FN_SINPI = 4,
  FN_COSPI = 5,
  FN_TANH  = 6,
  FN_COUNT = 7
};

constexpr uint32_t QNAN_F32 = 0x7fc00000;

uint32_t eval(uint32_t fn, uint32_t x);

const char* fn_name(uint32_t fn);

} // namespace vortex::mufu
