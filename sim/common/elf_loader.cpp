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

#include "elf_loader.h"
#include "mem.h"
#include <elf.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace vortex;

namespace {

// Type bundle that lets one parser body cover ELF32 and ELF64.
struct Elf32Traits {
  using Ehdr = Elf32_Ehdr;
  using Phdr = Elf32_Phdr;
  using Shdr = Elf32_Shdr;
  using Sym  = Elf32_Sym;
};
struct Elf64Traits {
  using Ehdr = Elf64_Ehdr;
  using Phdr = Elf64_Phdr;
  using Shdr = Elf64_Shdr;
  using Sym  = Elf64_Sym;
};

// Zero a [addr, addr+size) region of RAM in bounded chunks.
void zero_fill(RAM& ram, uint64_t addr, uint64_t size) {
  static const std::vector<uint8_t> zeros(64 * 1024, 0);
  while (size > 0) {
    uint64_t chunk = std::min<uint64_t>(size, zeros.size());
    ram.write(zeros.data(), addr, chunk);
    addr += chunk;
    size -= chunk;
  }
}

template <typename T>
bool parse_elf(const std::vector<uint8_t>& buf, RAM& ram, ElfImage* out) {
  using Ehdr = typename T::Ehdr;
  using Phdr = typename T::Phdr;
  using Shdr = typename T::Shdr;
  using Sym  = typename T::Sym;

  if (buf.size() < sizeof(Ehdr)) {
    std::cerr << "Error: ELF file truncated (header)" << std::endl;
    return false;
  }
  const auto* eh = reinterpret_cast<const Ehdr*>(buf.data());

  if (eh->e_machine != EM_RISCV) {
    std::cerr << "Error: ELF is not a RISC-V executable (e_machine="
              << eh->e_machine << ")" << std::endl;
    return false;
  }

  out->entry = eh->e_entry;

  // Copy PT_LOAD segments to their physical addresses.
  for (unsigned i = 0; i < eh->e_phnum; ++i) {
    uint64_t off = uint64_t(eh->e_phoff) + uint64_t(i) * eh->e_phentsize;
    if (off + sizeof(Phdr) > buf.size()) {
      std::cerr << "Error: ELF program header out of range" << std::endl;
      return false;
    }
    const auto* ph = reinterpret_cast<const Phdr*>(buf.data() + off);
    if (ph->p_type != PT_LOAD || ph->p_memsz == 0)
      continue;
    if (uint64_t(ph->p_offset) + ph->p_filesz > buf.size()) {
      std::cerr << "Error: ELF segment out of range" << std::endl;
      return false;
    }
    if (ph->p_filesz > 0) {
      ram.write(buf.data() + ph->p_offset, ph->p_paddr, ph->p_filesz);
    }
    // Zero-fill the BSS tail (p_memsz > p_filesz).
    if (ph->p_memsz > ph->p_filesz) {
      zero_fill(ram, uint64_t(ph->p_paddr) + ph->p_filesz,
                uint64_t(ph->p_memsz) - ph->p_filesz);
    }
  }

  // Resolve the HTIF `tohost` symbol from the symbol table, if present.
  for (unsigned i = 0; i < eh->e_shnum; ++i) {
    uint64_t off = uint64_t(eh->e_shoff) + uint64_t(i) * eh->e_shentsize;
    if (off + sizeof(Shdr) > buf.size())
      break;
    const auto* sh = reinterpret_cast<const Shdr*>(buf.data() + off);
    if (sh->sh_type != SHT_SYMTAB)
      continue;
    // The string table for this symbol table is referenced by sh_link.
    uint64_t str_off = uint64_t(eh->e_shoff) + uint64_t(sh->sh_link) * eh->e_shentsize;
    if (str_off + sizeof(Shdr) > buf.size())
      break;
    const auto* strsh = reinterpret_cast<const Shdr*>(buf.data() + str_off);
    const char* strtab = reinterpret_cast<const char*>(buf.data() + strsh->sh_offset);
    uint64_t strtab_end = uint64_t(strsh->sh_offset) + strsh->sh_size;
    if (strtab_end > buf.size() || sh->sh_entsize == 0)
      break;

    uint64_t count = sh->sh_size / sh->sh_entsize;
    for (uint64_t s = 0; s < count; ++s) {
      uint64_t sym_off = uint64_t(sh->sh_offset) + s * sh->sh_entsize;
      if (sym_off + sizeof(Sym) > buf.size())
        break;
      const auto* sym = reinterpret_cast<const Sym*>(buf.data() + sym_off);
      uint64_t name_off = uint64_t(strsh->sh_offset) + sym->st_name;
      if (name_off >= strtab_end)
        continue;
      const char* name = strtab + sym->st_name;
      if (std::strcmp(name, "tohost") == 0) {
        out->has_tohost  = true;
        out->tohost_addr = sym->st_value;
        return true;
      }
    }
  }
  return true;
}

} // namespace

bool vortex::isElfFile(const char* filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs)
    return false;
  char magic[SELFMAG] = {0};
  ifs.read(magic, SELFMAG);
  return ifs.gcount() == SELFMAG
      && std::memcmp(magic, ELFMAG, SELFMAG) == 0;
}

bool vortex::loadElfImage(const char* filename, RAM& ram, ElfImage* out) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "Error: " << filename << " not found" << std::endl;
    return false;
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

  if (buf.size() < EI_NIDENT
   || std::memcmp(buf.data(), ELFMAG, SELFMAG) != 0) {
    std::cerr << "Error: " << filename << " is not an ELF file" << std::endl;
    return false;
  }

  *out = ElfImage{};
  ram.clear();

  switch (buf[EI_CLASS]) {
  case ELFCLASS32: return parse_elf<Elf32Traits>(buf, ram, out);
  case ELFCLASS64: return parse_elf<Elf64Traits>(buf, ram, out);
  default:
    std::cerr << "Error: unknown ELF class " << int(buf[EI_CLASS]) << std::endl;
    return false;
  }
}
