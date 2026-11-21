// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reserved and XLEN-dependent RVC code points.
//
// The `rvc` catalog runs compiled programs, so it only ever feeds the
// decompressor encodings an assembler chose to emit. Reserved code points are
// unreachable that way, which leaves the guards that reject them untested. This
// drives rvc_decompress directly with the raw half-words instead.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "decompressor.h"

using vortex::rvc_decompress;

namespace {

int failures = 0;

void expect_illegal(const char *what, uint32_t half) {
  auto r = rvc_decompress(half);
  if (!r.illegal) {
    printf("FAILED: %s (0x%04x) expanded to 0x%08x, expected illegal\n",
           what, half, r.instr32);
    ++failures;
  }
}

void expect_expansion(const char *what, uint32_t half, uint32_t expect) {
  auto r = rvc_decompress(half);
  if (r.illegal) {
    printf("FAILED: %s (0x%04x) reported illegal, expected 0x%08x\n",
           what, half, expect);
    ++failures;
  } else if (r.instr32 != expect) {
    printf("FAILED: %s (0x%04x) expanded to 0x%08x, expected 0x%08x\n",
           what, half, r.instr32, expect);
    ++failures;
  }
}

} // namespace

int main() {
  // Shift-immediate code points with shamt[5] set. The C extension reserves
  // these for custom extensions when XLEN=32 — expanding them there yields a
  // shift by 32 or more, which is itself an illegal RV32 encoding.
  //   c.slli x1, 32   c.srli x8, 32   c.srai x8, 32
#ifdef VX_CFG_XLEN_64
  expect_expansion("c.slli shamt[5]=1", 0x1082, 0x02009093);
  expect_expansion("c.srli shamt[5]=1", 0x9001, 0x02045413);
  expect_expansion("c.srai shamt[5]=1", 0x9401, 0x42045413);
#else
  expect_illegal("c.slli shamt[5]=1", 0x1082);
  expect_illegal("c.srli shamt[5]=1", 0x9001);
  expect_illegal("c.srai shamt[5]=1", 0x9401);
#endif

  // The same instructions with shamt[5] clear expand identically at both widths.
  expect_expansion("c.slli shamt=1", 0x0086, 0x00109093);
  expect_expansion("c.srli shamt=1", 0x8005, 0x00145413);
  expect_expansion("c.srai shamt=1", 0x8405, 0x40145413);

  // C.SUBW/C.ADDW are RV64-only; the RV32 guard here is the precedent the
  // shamt[5] guard above follows.
#ifdef VX_CFG_XLEN_64
  expect_expansion("c.subw", 0x9d05, 0x4095053b);
  expect_expansion("c.addw", 0x9d25, 0x0095053b);
#else
  expect_illegal("c.subw", 0x9d05);
  expect_illegal("c.addw", 0x9d25);
#endif

  // Code points with rd=x0 are HINTs: they expand to a write of x0.
  expect_expansion("c.slli rd=x0", 0x0002, 0x00001013);
  expect_expansion("c.lui rd=x0", 0x6005, 0x00001037);
  expect_expansion("c.mv rd=x0", 0x8006, 0x00100033);
  expect_expansion("c.add rd=x0", 0x9006, 0x00100033);

  // Reserved code points.
  expect_illegal("c.lui imm=0", 0x6081);
  expect_illegal("c.addi16sp imm=0", 0x6101);
  expect_illegal("c.jr rs1=x0", 0x8002);

  // C.FLD/C.FSD and their SP forms exist only with 64-bit FP registers.
#ifdef VX_CFG_FLEN_64
  expect_expansion("c.fld", 0x2000, 0x00043407);
#else
  expect_illegal("c.fld", 0x2000);
  expect_illegal("c.fsd", 0xa000);
  expect_illegal("c.fldsp", 0x2002);
  expect_illegal("c.fsdsp", 0xa002);
#endif

  if (failures != 0) {
    printf("FAILED: %d rvc_decompress case(s)\n", failures);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
