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

#include <mempool.h>
#include <simobject.h>
#include "instr.h"

namespace vortex {

// Stateless ISA decoder.
class Decoder : public SimObject<Decoder> {
public:
  Decoder(const SimContext& ctx, const char* name, PoolAllocator<Instr, 64>& instr_pool);
  ~Decoder();

  // Decode an instruction word into an Instr. `code` carries either a
  // 32-bit RV32I encoding (low2 == 0b11) or a 16-bit RVC in the low half
  // (low2 != 0b11); the decoder detects RVC, calls rvc_decompress, and
  // stamps IntrBrArgs.is_rvc accordingly.
  Instr::Ptr decode(uint32_t code, uint64_t uuid);

private:
  PoolAllocator<Instr, 64>& instr_pool_;
};

}
