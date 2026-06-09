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

#include "host_monitor.h"
#include "mem.h"

using namespace vortex;

bool HostMonitor::tick(RAM& ram) {
  if (!enabled_ || terminated_)
    return terminated_;
  // tohost is a 64-bit word. riscv-tests' .tohost section is zero-
  // initialized, so the high half stays 0 on RV32 (only `sw` writes the
  // low half) and a plain 8-byte read is unambiguous.
  uint64_t v = 0;
  ram.read(&v, tohost_addr_, sizeof(uint64_t));
  if (v != 0) {
    captured_   = v;
    terminated_ = true;
  }
  return terminated_;
}

int HostMonitor::exit_code() const {
  if (!terminated_)
    return -1;
  // bit 0 must be set for a well-formed halt request.
  if ((captured_ & 1) == 0)
    return -1;
  return static_cast<int>(captured_ >> 1);
}
