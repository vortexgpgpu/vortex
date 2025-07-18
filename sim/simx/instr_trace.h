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
  LsuTraceData(uint32_t num_threads = 0) : mem_addrs(num_threads) {}
};

struct SfuTraceData : public ITraceData {
  using Ptr = std::shared_ptr<SfuTraceData>;
  Word arg1;
  Word arg2;
  SfuTraceData(Word arg1, Word arg2) : arg1(arg1), arg2(arg2) {}
};

struct instr_trace_t {
public:
  //--
  const uint64_t uuid;
  const Arch& arch;

  //--
  uint32_t    cid;
  uint32_t    wid;
  ThreadMask  tmask;
  Word        PC;
  bool        wb;

  //--
  RegOpd      dst_reg;

  //--
  std::vector<RegOpd> src_regs;

  //-
  FUType     fu_type;

  //--
  OpType     op_type;

  ITraceData::Ptr data;

  int  pid;
  bool sop;
  bool eop;

  bool fetch_stall;

  uint64_t issue_time ;

  instr_trace_t(uint64_t uuid, const Arch& arch)
    : uuid(uuid)
    , arch(arch)
    , cid(0)
    , wid(0)
    , tmask(0)
    , PC(0)
    , wb(false)
    , dst_reg({RegType::None, 0})
    , src_regs(NUM_SRC_REGS, {RegType::None, 0})
    , fu_type(FUType::ALU)
    , op_type({})
    , data(nullptr)
    , pid(-1)
    , sop(true)
    , eop(true)
    , fetch_stall(false)
    , issue_time(SimPlatform::instance().cycles())
    , log_once_(false)
  {}

  instr_trace_t(const instr_trace_t& rhs)
    : uuid(rhs.uuid)
    , arch(rhs.arch)
    , cid(rhs.cid)
    , wid(rhs.wid)
    , tmask(rhs.tmask)
    , PC(rhs.PC)
    , wb(rhs.wb)
    , dst_reg(rhs.dst_reg)
    , src_regs(rhs.src_regs)
    , fu_type(rhs.fu_type)
    , op_type(rhs.op_type)
    , data(rhs.data)
    , pid(rhs.pid)
    , sop(rhs.sop)
    , eop(rhs.eop)
    , fetch_stall(rhs.fetch_stall)
    , issue_time(rhs.issue_time)
    , log_once_(false)
  {}

  ~instr_trace_t() {}

  bool log_once(bool enable) {
    bool old = log_once_;
    log_once_ = enable;
    return old;
  }

  friend std::ostream &operator<<(std::ostream &os, const instr_trace_t& trace) {
    os << "cid=" << trace.cid;
    os << ", wid=" << trace.wid;
    os << ", tmask=";
    for (uint32_t i = 0, n = trace.arch.num_threads(); i < n; ++i) {
      os << trace.tmask.test(i);
    }
    os << ", PC=0x" << std::hex << trace.PC << std::dec;
    os << ", wb=" << trace.wb;
    if (trace.dst_reg.type != RegType::None) {
      os << ", rd=" << trace.dst_reg;
    }
    for (uint32_t i = 0; i < trace.src_regs.size(); ++i) {
      if (trace.src_regs[i].type != RegType::None) {
        os << ", rs" << i << "=" << trace.src_regs[i];
      }
    }
    os << ", ex=" << trace.fu_type;
    if (trace.pid != -1) {
      os << ", pid=" << trace.pid;
      os << ", sop=" << trace.sop;
      os << ", eop=" << trace.eop;
    }
    os << " (#" << trace.uuid << ")";
    return os;
  }

private:
  bool log_once_;
};

using TraceArbiter = TxArbiter<instr_trace_t*>;

}