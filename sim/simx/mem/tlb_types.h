// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include <VX_types.h>
#include <cstdint>
#include <ostream>

namespace vortex {

// VPN bits consumed per page-table level: the page-table fan-out, which
// halves between Sv32 and Sv39 because the PTE doubles in width.
constexpr uint32_t tlb_vpn_level_bits(uint32_t fanout) {
  return (fanout <= 1) ? 0 : 1 + tlb_vpn_level_bits(fanout / 2);
}
constexpr uint32_t TLB_VPN_LEVEL_BITS =
  tlb_vpn_level_bits(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);

enum class TlbAccess : uint8_t {
  Read  = 0,
  Write = 1,
  Exec  = 2
};

// Permission test against a leaf PTE's flag bits. Kernels run in U-mode,
// so U is always required; W covers stores and the write half of an AMO.
// Shared by the walker and by every TLB that hands out a translation, so
// the two can never disagree about what a page permits.
inline bool tlb_perm_ok(uint8_t flags, TlbAccess access, bool amo) {
  const bool r = (flags & (1u << 1)) != 0;
  const bool w = (flags & (1u << 2)) != 0;
  const bool x = (flags & (1u << 3)) != 0;
  const bool u = (flags & (1u << 4)) != 0;
  if (!u) {
    return false;
  }
  if (access == TlbAccess::Exec) {
    return x;
  }
  if (access == TlbAccess::Write || amo) {
    return w;
  }
  return r;
}

// Miss request from an L1 TLB (or client uTLB) to the shared cluster TLB.
struct TlbReq {
  uint64_t  vpn = 0;
  TlbAccess access = TlbAccess::Read;
  bool      amo = false;      // AMO write intent (carried with rw=0 requests)
  uint32_t  client_id = 0;    // requesting L1 instance, for fill routing
  uint32_t  slot = 0;         // requester's miss-station slot
  // A permission violation caught against an already-cached translation:
  // carries no walk and expects no fill, it only reaches the walker so the
  // fault is latched where the host reads it.
  bool      report_only = false;
};

// Fill (or fault) response back to the requesting L1.
struct TlbRsp {
  uint64_t ppn = 0;
  uint8_t  flags = 0;
  uint8_t  level = 0;         // 0 = base page, 1 = mega, 2 = giga
  bool     fault = false;     // no translation installed when set
  uint32_t client_id = 0;
  uint32_t slot = 0;
};

inline std::ostream& operator<<(std::ostream& os, const TlbReq& req) {
  os << "vpn=0x" << std::hex << req.vpn << std::dec
     << ", access=" << (int)req.access << ", amo=" << req.amo
     << ", client=" << req.client_id << ", slot=" << req.slot;
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TlbRsp& rsp) {
  os << "ppn=0x" << std::hex << rsp.ppn << std::dec
     << ", level=" << (int)rsp.level << ", fault=" << rsp.fault
     << ", client=" << rsp.client_id << ", slot=" << rsp.slot;
  return os;
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
