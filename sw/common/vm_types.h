// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// RISC-V Sv32/Sv39 SATP / PTE / vAddr value classes. Pure host-side
// arithmetic, shared by the simulator MMU model and the host runtime.
// Mode selection follows VX_VM_ADDR_MODE from VX_types.h.

#pragma once

#include <VX_types.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <cassert>

namespace vortex {

#define BARE 0x0
#define SV32 0x1
#define SV39 0x8

enum ACCESS_TYPE {
  LOAD,
  STORE,
  FETCH
};

class SATP_t {
private:
  uint64_t address;
  uint16_t asid;
  uint8_t  mode;
  uint64_t ppn;
  uint64_t satp;

  static uint64_t bits(uint64_t input, uint8_t s_idx, uint8_t e_idx) {
    return (input >> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
  }
  static bool bit(uint64_t input, uint8_t idx) {
    return (input) & ((uint64_t)1 << idx);
  }

public:
  SATP_t(uint64_t satp_val) : satp(satp_val) {
#if VX_VM_ADDR_MODE == SV32
    mode = bit(satp, 31);
    asid = bits(satp, 22, 30);
    ppn  = bits(satp, 0, 21);
#else
    mode = bits(satp, 60, 63);
    asid = bits(satp, 44, 59);
    ppn  = bits(satp, 0, 43);
#endif
    address = ppn << VX_VM_PAGE_LOG2_SIZE;
  }

  SATP_t(uint64_t address, uint16_t asid) : address(address), asid(asid) {
#if VX_VM_ADDR_MODE == SV32
    assert((address >> 32) == 0 && "Upper 32 bits are not zero!");
#endif
    mode = VX_VM_ADDR_MODE;
    ppn = address >> VX_VM_PAGE_LOG2_SIZE;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#if VX_VM_ADDR_MODE == SV32
    satp = (((uint64_t)mode << 31) | ((uint64_t)asid << 22) | ppn);
#else
    satp = (((uint64_t)mode << 60) | ((uint64_t)asid << 44) | ppn);
#endif
#pragma GCC diagnostic pop
  }

  uint8_t  get_mode()     const { return mode; }
  uint16_t get_asid()     const { return asid; }
  uint64_t get_base_ppn() const { return ppn; }
  uint64_t get_satp()     const { return satp; }
};

class Page_Fault_Exception : public std::runtime_error {
public:
  Page_Fault_Exception(const std::string& what = "") : std::runtime_error(what) {}
  uint64_t addr;
  ACCESS_TYPE type;
};

class PTE_t {
private:
  uint64_t address;
  static uint64_t bits(uint64_t input, uint8_t s_idx, uint8_t e_idx) {
    return (input >> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
  }
  static bool bit(uint64_t input, uint8_t idx) {
    return (input) & ((uint64_t)1 << idx);
  }

public:
#if VX_VM_ADDR_MODE == SV39
  bool N;
  uint8_t PBMT;
#endif
  uint64_t ppn;
  uint32_t rsw;
  uint32_t flags;
  uint8_t level;
  bool d, a, g, u, x, w, r, v;
  uint64_t pte_bytes;

  void set_flags(uint32_t flag) {
    this->flags = flag;
    d = bit(flags, 7);
    a = bit(flags, 6);
    g = bit(flags, 5);
    u = bit(flags, 4);
    x = bit(flags, 3);
    w = bit(flags, 2);
    r = bit(flags, 1);
    v = bit(flags, 0);
  }

  PTE_t(uint64_t address, uint32_t flags) : address(address) {
#if VX_VM_ADDR_MODE == SV39
    N = 0;
    PBMT = 0;
    level = 3;
    ppn = address >> VX_VM_PAGE_LOG2_SIZE;
    set_flags(flags);
    pte_bytes = (ppn << 10) | flags;
#else // SV32
    assert((address >> 32) == 0 && "Upper 32 bits are not zero!");
    level = 2;
    ppn = address >> VX_VM_PAGE_LOG2_SIZE;
    set_flags(flags);
    pte_bytes = (ppn << 10) | flags;
#endif
  }

  PTE_t(uint64_t pte_bytes) : pte_bytes(pte_bytes) {
#if VX_VM_ADDR_MODE == SV39
    N = bit(pte_bytes, 63);
    PBMT = bits(pte_bytes, 61, 62);
    level = 3;
    ppn = bits(pte_bytes, 10, 53);
    address = ppn << VX_VM_PAGE_LOG2_SIZE;
#else // SV32
    assert((pte_bytes >> 32) == 0 && "Upper 32 bits are not zero!");
    level = 2;
    ppn = bits(pte_bytes, 10, 31);
    address = ppn << VX_VM_PAGE_LOG2_SIZE;
#endif
    rsw = bits(pte_bytes, 8, 9);
    set_flags((uint32_t)(bits(pte_bytes, 0, 7)));
  }
};

class vAddr_t {
private:
  uint64_t address;
  uint64_t bits(uint8_t s_idx, uint8_t e_idx) const {
    return (address >> s_idx) & (((uint64_t)1 << (e_idx - s_idx + 1)) - 1);
  }

public:
  uint64_t *vpn;
  uint64_t pgoff;
  uint8_t level;

  vAddr_t(uint64_t address) : address(address) {
#if VX_VM_ADDR_MODE == SV39
    level = 3;
    vpn = new uint64_t[level];
    vpn[2] = bits(30, 38);
    vpn[1] = bits(21, 29);
    vpn[0] = bits(12, 20);
    pgoff  = bits(0, 11);
#else // SV32
    assert((address >> 32) == 0 && "Upper 32 bits are not zero!");
    level = 2;
    vpn = new uint64_t[level];
    vpn[1] = bits(22, 31);
    vpn[0] = bits(12, 21);
    pgoff  = bits(0, 11);
#endif
  }

  ~vAddr_t() { delete[] vpn; }
};

} // namespace vortex
