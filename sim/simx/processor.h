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

#include <stdint.h>
#include <VX_config.h>
#include <mem.h>

namespace vortex {

class Arch;
class RAM;
class ProcessorImpl;
#ifdef VM_ENABLE
class SATP_t;
#endif

class Processor {
public:
  Processor(const Arch& arch);
  ~Processor();

  void attach_ram(RAM* mem);

  int run();

  void dcr_write(uint32_t addr, uint32_t value);
#ifdef VM_ENABLE
  bool is_satp_unset();
  uint8_t get_satp_mode();
  uint64_t get_base_ppn();
  int16_t set_satp_by_addr(uint64_t addr);
#endif

private:
  ProcessorImpl* impl_;
#ifdef VM_ENABLE
  SATP_t *satp_;
#endif
};

}
