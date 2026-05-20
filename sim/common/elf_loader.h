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

namespace vortex {

class RAM;

// Result of loading a RISC-V ELF executable into device RAM.
struct ElfImage {
  uint64_t entry       = 0;      // ELF entry point (e_entry)
  bool     has_tohost  = false;  // true if a `tohost` symbol was found
  uint64_t tohost_addr = 0;      // address of the `tohost` symbol, if any
};

// True if `filename` begins with the ELF magic. Lets the simulator
// dispatch on file content rather than extension — upstream riscv-tests
// produce extensionless ELF executables.
bool isElfFile(const char* filename);

// Load a RISC-V ELF executable into `ram`: copies every PT_LOAD segment
// to its physical address, zero-fills the BSS tail, resolves the entry
// point, and looks up the HTIF `tohost` symbol (used by riscv-tests for
// pass/fail signalling). Both ELF32 and ELF64 are supported.
//
// Returns true on success and fills `*out`. Returns false (and prints a
// diagnostic) on a malformed or non-RISC-V ELF.
bool loadElfImage(const char* filename, RAM& ram, ElfImage* out);

} // namespace vortex
