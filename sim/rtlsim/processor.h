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

namespace vortex {

class RAM;
class HostMonitor;
class CoutDrainer;

class Processor {
public:

  Processor();
  ~Processor();

  void attach_ram(RAM* ram);

  // When `monitor` is non-null and enabled, the run loop polls it each
  // cycle and stops as soon as the HTIF `tohost` word is written. When
  // `cout_drainer` is non-null, the run loop drains the lossless COUT
  // stream-ring every cycle — only standalone rtlsim wires this; the
  // runtime path leaves it null and drains COUT itself.
  void run(HostMonitor* monitor = nullptr, CoutDrainer* cout_drainer = nullptr);

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

private:

  class Impl;
  Impl* impl_;
};

}