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

#ifdef EXT_TMA_ENABLE
TmaDCRS::Descriptor TmaDCRS::read_descriptor(uint32_t slot) const {
  Descriptor desc{};
  desc.base_addr = (uint64_t(read_word(slot, VX_DCR_TMA_DESC_BASE_HI_OFF)) << 32)
                 | uint64_t(read_word(slot, VX_DCR_TMA_DESC_BASE_LO_OFF));

  desc.sizes = {
    read_word(slot, VX_DCR_TMA_DESC_SIZE0_OFF),
    read_word(slot, VX_DCR_TMA_DESC_SIZE1_OFF),
    read_word(slot, VX_DCR_TMA_DESC_SIZE2_OFF),
    read_word(slot, VX_DCR_TMA_DESC_SIZE3_OFF),
    read_word(slot, VX_DCR_TMA_DESC_SIZE4_OFF),
  };

  desc.strides = {
    read_word(slot, VX_DCR_TMA_DESC_STRIDE0_OFF),
    read_word(slot, VX_DCR_TMA_DESC_STRIDE1_OFF),
    read_word(slot, VX_DCR_TMA_DESC_STRIDE2_OFF),
    read_word(slot, VX_DCR_TMA_DESC_STRIDE3_OFF),
  };

  desc.meta = read_word(slot, VX_DCR_TMA_DESC_META_OFF);

  desc.element_strides = {
    read_word(slot, VX_DCR_TMA_DESC_ESTRIDE0_OFF),
    read_word(slot, VX_DCR_TMA_DESC_ESTRIDE1_OFF),
    read_word(slot, VX_DCR_TMA_DESC_ESTRIDE2_OFF),
    read_word(slot, VX_DCR_TMA_DESC_ESTRIDE3_OFF),
    read_word(slot, VX_DCR_TMA_DESC_ESTRIDE4_OFF),
  };

  uint32_t ts01 = read_word(slot, VX_DCR_TMA_DESC_TILESIZE01_OFF);
  uint32_t ts23 = read_word(slot, VX_DCR_TMA_DESC_TILESIZE23_OFF);
  uint32_t ts4 = read_word(slot, VX_DCR_TMA_DESC_TILESIZE4_OFF);
  desc.tile_sizes = {
    uint16_t(ts01 & 0xffff),
    uint16_t((ts01 >> 16) & 0xffff),
    uint16_t(ts23 & 0xffff),
    uint16_t((ts23 >> 16) & 0xffff),
    uint16_t(ts4 & 0xffff),
  };

  desc.cfill = read_word(slot, VX_DCR_TMA_DESC_CFILL_OFF);
  return desc;
}
#endif

void DCRS::write(uint32_t addr, uint32_t value) {
  if (addr >= VX_DCR_BASE_STATE_BEGIN
   && addr < VX_DCR_BASE_STATE_END) {
      base_dcrs.write(addr, value);
      return;
  }

#ifdef EXT_TMA_ENABLE
  if (addr >= VX_DCR_TMA_STATE_BEGIN
   && addr < VX_DCR_TMA_STATE_END) {
      tma_dcrs.write(addr, value);
      return;
  }
#endif

  std::cerr << "Error: invalid global DCR addr=0x" << std::hex << addr << std::dec << std::endl;
  std::abort();
}
