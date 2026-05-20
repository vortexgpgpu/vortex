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

#include <cstdint>
#include "elf_loader.h"

namespace vortex {

class RAM;

// HTIF `tohost` exit watcher for upstream riscv-tests/isa.
//
// The ISA tests signal pass/fail by storing `(code << 1) | 1` to the
// `tohost` memory word. Vortex's L1 dcache is write-through, so the
// store reaches device RAM within a bounded latency and the monitor
// observes it by polling.
//
// The monitor stays disabled when the loaded image has no `tohost`
// symbol — normal Vortex kernels and the (MMIO-based) benchmarks are
// unaffected and terminate through the usual IO_EXIT_CODE path.
class HostMonitor {
public:
  HostMonitor() = default;

  // Bind to a loaded image. Enabled only if the image exposes `tohost`.
  void attach(const ElfImage& img) {
    enabled_     = img.has_tohost;
    tohost_addr_ = img.tohost_addr;
  }

  // Poll the `tohost` word. Returns true once it has gone non-zero.
  bool tick(RAM& ram);

  bool enabled() const    { return enabled_; }
  bool terminated() const { return terminated_; }

  // Spike/riscv-tests encoding: bit 0 set means "halt"; the exit code is
  // the value shifted right by one (0 = pass, N = failed subtest N).
  int exit_code() const;

private:
  bool     enabled_    = false;
  bool     terminated_ = false;
  uint64_t tohost_addr_ = 0;
  uint64_t captured_    = 0;
};

} // namespace vortex
