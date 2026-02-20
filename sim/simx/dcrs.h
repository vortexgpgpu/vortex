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

#include <util.h>
#include <VX_types.h>
#include <array>
#include "constants.h"

namespace vortex {

class BaseDCRS {
public:
  uint32_t read(uint32_t addr) const {
    uint32_t state = VX_DCR_BASE_STATE(addr);
    return states_.at(state);
  }

	void write(uint32_t addr, uint32_t value) {
		uint32_t state = VX_DCR_BASE_STATE(addr);
		states_.at(state) = value;
	}

private:
  std::array<uint32_t, VX_DCR_BASE_STATE_COUNT> states_;
};

#ifdef EXT_TMA_ENABLE
class TmaDCRS {
public:
  struct Descriptor {
    uint64_t base_addr;
    std::array<uint32_t, 5> sizes;
    std::array<uint32_t, 4> strides;
    uint32_t meta;
    std::array<uint32_t, 5> element_strides;
    std::array<uint16_t, 5> tile_sizes;
    uint32_t cfill;
  };

  void write(uint32_t addr, uint32_t value) {
    uint32_t slot = VX_DCR_TMA_DESC_SLOT(addr);
    uint32_t word = VX_DCR_TMA_DESC_WORD(addr);
    states_.at(slot).at(word) = value;
  }

  uint32_t read_word(uint32_t slot, uint32_t word) const {
    return states_.at(slot).at(word);
  }

  Descriptor read_descriptor(uint32_t slot) const;

private:
  std::array<std::array<uint32_t, VX_DCR_TMA_DESC_STRIDE>, VX_DCR_TMA_DESC_COUNT> states_;
};
#endif

class DCRS {
public:
  void write(uint32_t addr, uint32_t value);

  BaseDCRS base_dcrs;
#ifdef EXT_TMA_ENABLE
  TmaDCRS tma_dcrs;
#endif
};

}
