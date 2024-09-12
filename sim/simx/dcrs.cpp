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

#include "dcrs.h"
#include <iostream>

using namespace vortex;

void DCRS::write(uint32_t addr, uint32_t value) {
  if (addr >= VX_DCR_BASE_STATE_BEGIN
   && addr < VX_DCR_BASE_STATE_END) {
      base_dcrs.write(addr, value);
      return;
  }

  std::cout << "Error: invalid global DCR addr=0x" << std::hex << addr << std::dec << std::endl;
  std::abort();
}