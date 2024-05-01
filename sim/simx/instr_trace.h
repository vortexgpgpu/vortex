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

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "debug.h"

namespace vortex {

class ITraceData {
public:
    using Ptr = std::shared_ptr<ITraceData>;
    ITraceData() {}
    virtual ~ITraceData() {}
};

struct LsuTraceData : public ITraceData {
  using Ptr = std::shared_ptr<LsuTraceData>;
  std::vector<mem_addr_size_t> mem_addrs;
  LsuTraceData(uint32_t num_threads) : mem_addrs(num_threads) {}
};

struct SFUTraceData : public ITraceData {
  using Ptr = std::shared_ptr<SFUTraceData>;
  Word arg1;
  Word arg2;
  SFUTraceData(Word arg1, Word arg2) : arg1(arg1), arg2(arg2) {}
};

struct instr_trace_t {
public:
  //--
  const uint64_t uuid;
  const Arch&    arch;

  //--
  uint32_t    cid;
  uint32_t    wid;
  ThreadMask  tmask;
  Word        PC;

  //--
  uint32_t    rdest;
  RegType     rdest_type;
  bool        wb;

  //--
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //-
  FUType     fu_type;

  //--
  union {
    uint32_t unit_type;
    LsuType  lsu_type;
    AluType  alu_type;
    FpuType  fpu_type;
    SfuType  sfu_type;
  };

  ITraceData::Ptr data;

  int pid;
  bool sop;
  bool eop;

  bool fetch_stall;

  instr_trace_t(uint64_t uuid, const Arch& arch)
    : uuid(uuid)
    , arch(arch)
    , cid(0)
    , wid(0)
    , tmask(0)
    , PC(0)
    , rdest(0)
    , rdest_type(RegType::None)
    , wb(false)
    , used_iregs(0)
    , used_fregs(0)
    , used_vregs(0)
    , fu_type(FUType::ALU)
    , unit_type(0)
    , data(nullptr)
    , pid(-1)
    , sop(true)
    , eop(true)
    , fetch_stall(false)
    , log_once_(false)
  {}

  instr_trace_t(const instr_trace_t& rhs)
    : uuid(rhs.uuid)
    , arch(rhs.arch)
    , cid(rhs.cid)
    , wid(rhs.wid)
    , tmask(rhs.tmask)
    , PC(rhs.PC)
    , rdest(rhs.rdest)
    , rdest_type(rhs.rdest_type)
    , wb(rhs.wb)
    , used_iregs(rhs.used_iregs)
    , used_fregs(rhs.used_fregs)
    , used_vregs(rhs.used_vregs)
    , fu_type(rhs.fu_type)
    , unit_type(rhs.unit_type)
    , data(rhs.data)
    , pid(rhs.pid)
    , sop(rhs.sop)
    , eop(rhs.eop)
    , fetch_stall(rhs.fetch_stall)
    , log_once_(false)
  {}

  ~instr_trace_t() {}

  bool log_once(bool enable) {
    bool old = log_once_;
    log_once_ = enable;
    return old;
  }

private:
  bool log_once_;
};

inline std::ostream &operator<<(std::ostream &os, const instr_trace_t& trace) {
  os << "cid=" << trace.cid;
  os << ", wid=" << trace.wid;
  os << ", tmask=";
  for (uint32_t i = 0, n = trace.arch.num_threads(); i < n; ++i) {
      os << trace.tmask.test(i);
  }
  os << ", PC=0x" << std::hex << trace.PC;
  os << ", wb=" << trace.wb;
  if (trace.wb) {
     os << ", rd=" << trace.rdest_type << std::dec << trace.rdest;
  }
  os << ", ex=" << trace.fu_type;
  if (trace.pid != -1) {
    os << ", pid=" << trace.pid;
    os << ", sop=" << trace.sop;
    os << ", eop=" << trace.eop;
  }
  os << " (#" << std::dec << trace.uuid << ")";
  return os;
}

}