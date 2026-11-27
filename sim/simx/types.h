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
#include <algorithm>
#include <array>
#include <bitset>
#include <memory>
#include <queue>
#include <vector>
#include <unordered_map>
#include <variant>
#include <functional>
#include <util.h>
#include <stringutil.h>
#include <VX_config.h>
#include <VX_types.h>

// Simulation-kernel selection for the SimX build. The component-facing API
// (SimObject, SimPortIn/SimPortOut, SimEventLink, SimPlatform) is identical in both
// kernels; -DSIMX_FUNCTIONAL (set via build CONFIGS, uniform across every
// object in a build) selects the functional kernel, which disables timing
// simulation (unit latency, no back-pressure) for full-speed architectural
// runs. Each kernel's symbols differ at link time, so mixing objects built
// against different kernels fails loudly instead of violating ODR.
#include <simobject_lib.h>
#include <cstring>
#include <deque>
#include <map>
#include <sstream>
#include <linked_list.h>
#include <mempool.h>
#include <ringqueue.h>

#include <bitvector.h>
#include <fixed_vector.h>
#include <iostream>
#include "debug.h"

// Worker-thread count for the lockstep executor, set by the build system
// (e.g. CONFIGS="-DSIMX_MT=4"); 0 or 1 selects serial execution. Resolved
// here into a typed constant and handed to the kernel at platform setup —
// the kernel headers themselves are knob-free.
#ifndef SIMX_MT
#define SIMX_MT 0
#endif
// Synchronization quantum of the functional kernel, in cycles (e.g.
// CONFIGS="-DSIMX_FUNCTIONAL -DSIMX_MT=8 -DSIMX_QUANTUM=100"); 1 keeps
// per-cycle lockstep. The timed kernel accepts only 1.
#ifndef SIMX_QUANTUM
#define SIMX_QUANTUM 1
#endif
#include "constants.h"

namespace vortex {

constexpr uint32_t SIMX_NUM_WORKERS = (SIMX_MT > 1) ? SIMX_MT : 1;
constexpr uint32_t SIMX_QUANTUM_CYCLES = (SIMX_QUANTUM > 1) ? SIMX_QUANTUM : 1;

// One memory block (a cache line / DRAM transfer unit). Carried by
// MemReq/MemRsp in TLM data-path mode. Use `shared_ptr<mem_block_t>` so
// MSHR-coalesced replays share a single fill buffer without copying.
using mem_block_t = std::array<uint8_t, VX_CFG_MEM_BLOCK_SIZE>;

typedef uint8_t Byte;

#if (VX_CFG_XLEN == 32)
typedef uint32_t Word;
typedef int32_t  WordI;
typedef uint64_t DWord;
typedef int64_t  DWordI;
#elif (VX_CFG_XLEN == 64)
typedef uint64_t Word;
typedef int64_t  WordI;
typedef __uint128_t DWord;
typedef __int128_t DWordI;
#else
#error unsupported VX_CFG_XLEN
#endif

typedef std::bitset<MAX_NUM_CORES>   CoreMask;
typedef std::bitset<MAX_NUM_REGS>    RegMask;
typedef BitVector<Word>              ThreadMask;
typedef std::bitset<MAX_NUM_WARPS>   WarpMask;

///////////////////////////////////////////////////////////////////////////////

// Decode a packed kernel bar_id (local_group_id | (bar_no << 8)) to a flat barrier index.
// Preserves the global barrier flag (bit 31).
inline uint32_t bar_decode_id(uint32_t bar_id_raw, uint32_t num_barriers) {
  uint32_t cta_no = bar_id_raw & 0xffu;
  uint32_t bar_no = (bar_id_raw >> 8) & 0x7fffffu;
  return (cta_no * num_barriers + bar_no) | (bar_id_raw & 0x80000000u);
}

///////////////////////////////////////////////////////////////////////////////

union reg_data_t {
  uint8_t  u8;
  uint16_t u16;
  Word     u;
  WordI    i;
  float    f32;
  double   f64;
  uint32_t u32;
  uint64_t u64;
  int32_t  i32;
  int64_t  i64;
};

// One register value per thread: inline storage sized by the build's thread
// count, so traces and operand reads never touch the heap.
using RegDataVec = FixedVector<reg_data_t, VX_CFG_NUM_THREADS>;

///////////////////////////////////////////////////////////////////////////////

struct warp_barrier_t {
  WarpMask wait_mask    = 0;  // warps actually suspended on this barrier
  WarpMask arrival_mask = 0;  // global-barrier per-core arrival tracker
  uint32_t count  = 0;
  uint32_t events = 0;
  uint32_t phase  = 0;

  void reset() {
    wait_mask.reset();
    arrival_mask.reset();
    count  = 0;
    events = 0;
    phase  = 0;
  }
};

struct core_barrier_t {
  CoreMask mask = 0;

  void reset() {
    mask = 0;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct op_string_t {
  std::string op;
  std::string arg;
};

///////////////////////////////////////////////////////////////////////////////

enum class RegType {
  None,
  Integer,
  Float,
  Count
};

inline std::ostream &operator<<(std::ostream &os, const RegType& type) {
  switch (type) {
  case RegType::None: break;
  case RegType::Integer: os << "x"; break;
  case RegType::Float:   os << "f"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct RegOpd {
  RegType type = RegType::None;
  uint32_t idx = 0;

  uint32_t id() const {
    if (type == RegType::None)
      return 0;
    // unique register id embedding the type
    return (((int)(type)-1) << LOG_NUM_REGS) | idx;
  }

  friend std::ostream &operator<<(std::ostream &os, const RegOpd& reg) {
    os << reg.type << reg.idx;
    return os;
  }

  constexpr static uint32_t ID_BITS = log2ceil((int)RegType::Count) + LOG_NUM_REGS;
};

///////////////////////////////////////////////////////////////////////////////

enum class FUType {
  ALU,
  LSU,
  FPU,
  SFU,
#ifdef VX_CFG_EXT_TCU_ENABLE
  TCU,
#endif
  Count
};

inline std::ostream &operator<<(std::ostream &os, const FUType& type) {
  switch (type) {
  case FUType::ALU: os << "ALU"; break;
  case FUType::LSU: os << "LSU"; break;
  case FUType::FPU: os << "FPU"; break;
  case FUType::SFU: os << "SFU"; break;
#ifdef VX_CFG_EXT_TCU_ENABLE
  case FUType::TCU: os << "TCU"; break;
#endif
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AluType {
  LUI,
  AUIPC,
  ADD,
  SUB,
  SLL,
  SRL,
  SRA,
  SLT,
  SLTU,
  AND,
  OR,
  XOR,
  CZERO
};

struct IntrAluArgs {
  uint32_t is_imm : 1;
  uint32_t is_w : 1;
  uint32_t imm;
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  case AluType::LUI:     os << "LUI"; break;
  case AluType::AUIPC:   os << "AUIPC"; break;
  case AluType::ADD:     os << "ADD"; break;
  case AluType::SUB:     os << "SUB"; break;
  case AluType::SLL:     os << "SLL"; break;
  case AluType::SRL:     os << "SRL"; break;
  case AluType::SRA:     os << "SRA"; break;
  case AluType::SLT:     os << "SLT"; break;
  case AluType::SLTU:    os << "SLTU"; break;
  case AluType::AND:     os << "AND"; break;
  case AluType::OR:      os << "OR"; break;
  case AluType::XOR:     os << "XOR"; break;
  case AluType::CZERO:   os << "CZERO"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class BrType {
  BR,
  JAL,
  JALR,
  SYS,
  ILLEGAL,
};

struct IntrBrArgs {
  uint32_t cmp : 3;
  uint32_t is_rvc : 1;
  uint32_t offset;
};

inline std::ostream &operator<<(std::ostream &os, const BrType& type) {
  switch (type) {
  case BrType::BR:   os << "BR"; break;
  case BrType::JAL:  os << "JAL"; break;
  case BrType::JALR: os << "JALR"; break;
  case BrType::SYS:  os << "SYS"; break;
  case BrType::ILLEGAL: os << "ILLEGAL"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VoteType {
  ALL,
  ANY,
  UNI,
  BAL
};

inline std::ostream &operator<<(std::ostream &os, const VoteType& vote) {
  switch (vote) {
  case VoteType::ALL: os << "ALL"; break;
  case VoteType::ANY: os << "ANY"; break;
  case VoteType::UNI: os << "UNI"; break;
  case VoteType::BAL: os << "BAL"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class ShflType {
  UP,
  DOWN,
  BFLY,
  IDX
};

inline std::ostream &operator<<(std::ostream &os, const ShflType& shfl) {
  switch (shfl) {
  case ShflType::UP:   os << "UP"; break;
  case ShflType::DOWN: os << "DOWN"; break;
  case ShflType::BFLY: os << "BFLY"; break;
  case ShflType::IDX:  os << "IDX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class WgatherType {
  WGATHER
};

struct IntrWgatherArgs {
  uint32_t src_lane : 2;
};

inline std::ostream &operator<<(std::ostream &os, const WgatherType& /*type*/) {
  os << "WGATHER";
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class MdvType {
  MUL,
  MULHU,
  MULH,
  MULHSU,
  DIV,
  DIVU,
  REM,
  REMU
};

struct IntrMdvArgs {
  uint32_t is_w : 1;
};

inline std::ostream &operator<<(std::ostream &os, const MdvType& type) {
  switch (type) {
  case MdvType::MUL:    os << "MUL"; break;
  case MdvType::MULHU:  os << "MULHU"; break;
  case MdvType::MULH:   os << "MULH"; break;
  case MdvType::MULHSU: os << "MULHSU"; break;
  case MdvType::DIV:    os << "DIV"; break;
  case MdvType::DIVU:   os << "DIVU"; break;
  case MdvType::REM:    os << "REM"; break;
  case MdvType::REMU:   os << "REMU"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class LsuType {
  LOAD,
  STORE,
  FENCE
};

struct IntrLsuArgs {
  uint32_t width : 3;
  uint32_t stride : 4;  // per-uop multiplier: 0 = regular LD/ST; 0..N-1 = packLD uop_idx
  int32_t  offset;      // immediate addr offset (signed); 0 for packLD uops
  bool     matrix = false; // vx_ldm/vx_stm: offset[2:0] is the fragment shape
  uint8_t  fence_op = 0;   // a fence's operation (VX_FENCE_*)
};

inline std::ostream &operator<<(std::ostream &os, const LsuType& type) {
  switch (type) {
  case LsuType::LOAD:   os << "LOAD"; break;
  case LsuType::STORE:  os << "STORE"; break;
  case LsuType::FENCE:  os << "FENCE"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

// Memory-op tag. Always-present ops (LD/ST/FLUSH) at the low values,
// VX_CFG_EXT_A_ENABLE atomic family contiguous at 3..13.
enum class MemOp : uint8_t {
  // Always-present (independent of VX_CFG_EXT_A_ENABLE)
  LD        = 0,
  ST        = 1,
  FLUSH     = 2,
  // Atomic family (only meaningful when VX_CFG_EXT_A_ENABLE)
  AMO_LR    = 3,
  AMO_SC    = 4,
  AMO_SWAP  = 5,
  AMO_ADD   = 6,
  AMO_AND   = 7,
  AMO_OR    = 8,
  AMO_XOR   = 9,
  // MIN/MAX collapse signed/unsigned variants; the LSU sets
  // `flags.amo_unsigned` for AMOMINU/AMOMAXU and the AMO ALU branches on it.
  AMO_MIN   = 10,
  AMO_MAX   = 11,
  // Compare-and-swap. Kept last so the range checks below stay contiguous;
  // it is the only atomic that reads a third operand (the comparand, which
  // the instruction takes from rd).
  AMO_CAS   = 12,
};

inline bool memop_is_atomic(MemOp op) {
  return op >= MemOp::AMO_LR && op <= MemOp::AMO_CAS;
}
inline bool memop_is_amo_rmw(MemOp op) {
  return op >= MemOp::AMO_SWAP && op <= MemOp::AMO_CAS;
}
inline bool memop_is_write(MemOp op) {
  return op == MemOp::ST || op == MemOp::AMO_SC || memop_is_amo_rmw(op);
}
inline bool memop_is_amo(MemOp op) { return memop_is_atomic(op); }

// Memory-request flags.
struct MemFlags {
  union {
    uint32_t raw;
    struct {
      uint32_t strsp        : 1;  // bit 0: store-response enabled (opt-in)
      uint32_t io           : 1;  // bit 1: uncacheable I/O
      uint32_t local        : 1;  // bit 2: LMEM port
      uint32_t amo_unsigned : 1;  // bit 3: MIN/MAX signedness (1=unsigned)
      uint32_t flush        : 1;  // bit 4: flush token (a fence)
      uint32_t fence_op     : VX_FENCE_OP_BITS; // bits 5..8: a flush token's operation (VX_FENCE_*)
      uint32_t noalloc      : 1;  // bit 9: a write that misses a write-around cache goes to memory
    #ifdef VX_CFG_EXT_DXA_ENABLE
      // DXA completion sideband — populated only on DXA's LMEM-DMA writes.
      // notify_bar_id carries the RAW (kernel-encoded) bar handle:
      //   bits[7:0]   = cta_local_id (= get_local_group_id())
      //   bits[N+7:8] = user bar_no (N = CLOG2(NUM_BARRIERS))
      // Multicast adds cta_warp_idx into the low byte, so the field must
      // be wide enough to hold (bar_no << 8) | (MAX_CTAS_per_core - 1).
      // For NUM_BARRIERS = 8 and NUM_WARPS ≤ 16, that's at most 11 bits;
      // 16 bits leaves headroom for larger configs.
      uint32_t dxa_notify_done   : 1;   // bit 10
      uint32_t dxa_notify_bar_id : 16;  // bits 11..26
    #endif
    };
  };

  MemFlags() : raw(0) {}
  MemFlags(uint32_t r) : raw(r) {}

  bool any()  const { return raw != 0; }
  bool none() const { return raw == 0; }

  bool operator==(const MemFlags& o) const { return raw == o.raw; }
  bool operator!=(const MemFlags& o) const { return raw != o.raw; }

  friend std::ostream& operator<<(std::ostream& os, const MemFlags& f) {
    os << "0x" << std::hex << f.raw << std::dec;
    return os;
  }
};
static_assert(sizeof(MemFlags) == sizeof(uint32_t), "MemFlags must be 32-bit");

inline std::ostream &operator<<(std::ostream &os, const MemOp& op) {
  switch (op) {
  case MemOp::LD:        os << "LD"; break;
  case MemOp::ST:        os << "ST"; break;
  case MemOp::FLUSH:     os << "FLUSH"; break;
  case MemOp::AMO_LR:    os << "AMO_LR"; break;
  case MemOp::AMO_SC:    os << "AMO_SC"; break;
  case MemOp::AMO_SWAP:  os << "AMO_SWAP"; break;
  case MemOp::AMO_ADD:   os << "AMO_ADD"; break;
  case MemOp::AMO_AND:   os << "AMO_AND"; break;
  case MemOp::AMO_OR:    os << "AMO_OR"; break;
  case MemOp::AMO_XOR:   os << "AMO_XOR"; break;
  case MemOp::AMO_MIN:   os << "AMO_MIN"; break;
  case MemOp::AMO_MAX:   os << "AMO_MAX"; break;
  case MemOp::AMO_CAS:   os << "AMO_CAS"; break;
  default:               os << "?MemOp" << (int)op; break;
  }
  return os;
}

enum class AmoType {
  LR,
  SC,
  AMOADD,
  AMOSWAP,
  AMOAND,
  AMOOR,
  AMOXOR,
  AMOMIN,
  AMOMAX,
  AMOMINU,
  AMOMAXU,
  AMOCAS
};

struct IntrAmoArgs {
  uint32_t width : 3;
  uint32_t aq : 1;
  uint32_t rl : 1;
};

inline std::ostream &operator<<(std::ostream &os, const AmoType& type) {
  switch (type) {
  case AmoType::LR:      os << "LR"; break;
  case AmoType::SC:      os << "SC"; break;
  case AmoType::AMOADD:  os << "AMOADD"; break;
  case AmoType::AMOSWAP: os << "AMOSWAP"; break;
  case AmoType::AMOAND:  os << "AMOAND"; break;
  case AmoType::AMOOR:   os << "AMOOR"; break;
  case AmoType::AMOXOR:  os << "AMOXOR"; break;
  case AmoType::AMOMIN:  os << "AMOMIN"; break;
  case AmoType::AMOMAX:  os << "AMOMAX"; break;
  case AmoType::AMOMINU: os << "AMOMINU"; break;
  case AmoType::AMOMAXU: os << "AMOMAXU"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class FpuType {
  FADD,
  FSUB,
  FMUL,
  FDIV,
  FSQRT,
  FMADD,
  FMSUB,
  FNMADD,
  FNMSUB,
  F2I,
  I2F,
  F2F,
  FCMP,
  FSGNJ,
  FCLASS,
  FMVXW,
  FMVWX,
  FMINMAX,
};

struct IntrFpuArgs {
  uint32_t frm : 3;
  uint32_t cvt : 2;
  uint32_t is_f64 : 1;
  uint32_t is_f16 : 1;
};

inline std::ostream &operator<<(std::ostream &os, const FpuType& type) {
  switch (type) {
  case FpuType::FADD:   os << "FADD"; break;
  case FpuType::FSUB:   os << "FSUB"; break;
  case FpuType::FMUL:   os << "FMUL"; break;
  case FpuType::FDIV:   os << "FDIV"; break;
  case FpuType::FSQRT:  os << "FSQRT"; break;
  case FpuType::FMADD:  os << "FMADD"; break;
  case FpuType::FMSUB:  os << "FMSUB"; break;
  case FpuType::FNMADD: os << "FNMADD"; break;
  case FpuType::FNMSUB: os << "FNMSUB"; break;
  case FpuType::F2I:    os << "F2I"; break;
  case FpuType::I2F:    os << "I2F"; break;
  case FpuType::F2F:    os << "F2F"; break;
  case FpuType::FCMP:   os << "FCMP"; break;
  case FpuType::FSGNJ:  os << "FSGNJ"; break;
  case FpuType::FCLASS: os << "FCLASS"; break;
  case FpuType::FMVXW:  os << "FMVXW"; break;
  case FpuType::FMVWX:  os << "FMVWX"; break;
  case FpuType::FMINMAX: os << "FMIN_MAX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

#ifdef VX_CFG_EXT_TFU_ENABLE
// Values match the instruction's fn field and vortex::tfu::fn_t.
enum class TfuType {
  RCP   = 0,
  RSQRT = 1,
  EX2   = 2,
  LG2   = 3,
  SINPI = 4,
  COSPI = 5,
  TANH  = 6,
};

inline std::ostream &operator<<(std::ostream &os, const TfuType& type) {
  switch (type) {
  case TfuType::RCP:   os << "TFU.RCP"; break;
  case TfuType::RSQRT: os << "TFU.RSQRT"; break;
  case TfuType::EX2:   os << "TFU.EX2"; break;
  case TfuType::LG2:   os << "TFU.LG2"; break;
  case TfuType::SINPI: os << "TFU.SINPI"; break;
  case TfuType::COSPI: os << "TFU.COSPI"; break;
  case TfuType::TANH:  os << "TFU.TANH"; break;
  default:
    assert(false);
  }
  return os;
}
#endif

///////////////////////////////////////////////////////////////////////////////

enum class WctlType {
  TMC,
  WSPAWN,
  SPLIT,
  JOIN,
  BAR,
  PRED,
  WSYNC
};

struct IntrWctlArgs {
  uint32_t is_cond_neg : 1;
  uint32_t is_sync_bar : 1;
  uint32_t is_bar_arrive : 1;
};

inline std::ostream &operator<<(std::ostream &os, const WctlType& type) {
  switch (type) {
  case WctlType::TMC:    os << "TMC"; break;
  case WctlType::WSPAWN: os << "WSPAWN"; break;
  case WctlType::SPLIT:  os << "SPLIT"; break;
  case WctlType::JOIN:   os << "JOIN"; break;
  case WctlType::BAR:    os << "BAR"; break;
  case WctlType::PRED:   os << "PRED"; break;
  case WctlType::WSYNC:  os << "WSYNC"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

#ifdef VX_CFG_EXT_DXA_ENABLE

enum class DxaType {
  ISSUE
};

struct IntrDxaArgs {};

inline std::ostream &operator<<(std::ostream &os, const DxaType& type) {
  switch (type) {
  case DxaType::ISSUE: os << "DXA.ISSUE"; break;
  default: os << "?"; break;
  }
  return os;
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef VX_CFG_EXT_TEX_ENABLE

enum class TexType { SAMPLE };

struct IntrTexArgs {
  uint32_t stage : 2;     // texture stage (funct2)
};

inline std::ostream &operator<<(std::ostream &os, const TexType& type) {
  switch (type) {
  case TexType::SAMPLE: os << "TEX"; break;
  default: os << "?"; break;
  }
  return os;
}

#endif

#ifdef VX_CFG_EXT_OM_ENABLE

enum class OmType { EXPORT };

struct IntrOmArgs {
  // vx_om_export only: bit 0 = colour, bit 1 = depth. A shader emits colour only
  // (early-Z owns depth), depth only (z-prepass), or both (gl_FragDepth).
  uint32_t export_mask = 0;
};

inline std::ostream &operator<<(std::ostream &os, const OmType& type) {
  switch (type) {
  case OmType::EXPORT: os << "OM_EXPORT"; break;
  default: os << "?"; break;
  }
  return os;
}

#endif


#ifdef VX_CFG_EXT_RTU_ENABLE

// RTU (Ray-Tracing Unit) — PRISM ops.
// All share CUSTOM1 / funct3=5; sub-op (funct2) selects.
//   sub-op=0 SET    write one RTU reg (R-type, slot in funct7[6:2])
//   sub-op=1 GET    read one RTU reg
//   sub-op=2 TRACE  async ray issue; returns handle
//   sub-op=3 WAIT   block on handle; returns terminal status
//   sub-op=4 CB_RET release a yielded ray with an action code
//                   (ACCEPT / IGNORE / TERMINATE). rs1 = action.
enum class GfxwType {
  CB_RET,     // funct3=6 sub0: release a parked callback context
  TRACE,    // single-issue trace macro-op (expands to CFG/ORIGIN/DIR/ARM uops)
  WAIT,     // single-op block — parks until terminal, returns status
            // (parks/revives so it survives an async callback trap)
  GETWF,    // FP windowed regfile read (collapses N contiguous
            // float-slot vx_rt_get into one macro-op; callback read path)
  GETW,     // GP twin of GETWF (integer slots, no NaN-box). vx_rt_wait
            // reads t/u/v via GETWF and the IDs via GETW after the WAIT block.
};

struct IntrGfxwArgs {
  uint32_t slot  : 6;  // hit-window slot ID; GETWF: window start slot
  uint32_t uop   : 4;  // macro-op micro-op index (TRACE/WAIT/GETWF)
  uint32_t count : 4;  // GETWF: number of contiguous slots in the window (1..8)
};

inline std::ostream &operator<<(std::ostream &os, const GfxwType& type) {
  switch (type) {
  case GfxwType::CB_RET: os << "GFXW.CB_RET"; break;
  case GfxwType::TRACE: os << "GFXW.TRACE"; break;
  case GfxwType::WAIT:  os << "GFXW.WAIT";  break;
  case GfxwType::GETWF:  os << "GFXW.GETWF";  break;
  case GfxwType::GETW:   os << "GFXW.GETW";   break;
  default: os << "?"; break;
  }
  return os;
}

#endif

///////////////////////////////////////////////////////////////////////////////

enum class CsrType {
  CSRRW,
  CSRRS,
  CSRRC
};

struct IntrCsrArgs {
  uint32_t is_imm: 1;
  uint32_t imm : 5;
  uint32_t csr : 12;
};

inline std::ostream &operator<<(std::ostream &os, const CsrType& type) {
  switch (type) {
  case CsrType::CSRRW: os << "CSRRW"; break;
  case CsrType::CSRRS: os << "CSRRS"; break;
  case CsrType::CSRRC: os << "CSRRC"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class TcuType {
  WMMA,
  WGMMA,
  WMMA_SP,    // Sparse variants live in distinct op_types so IntrTcuArgs
  WGMMA_SP,   // doesn't carry a per-uop is_sparse bit.
  TCU_LD,     // Warp-level metadata load. rd[4] selects sparse/MX namespace.
};

struct IntrTcuArgs {
  uint32_t is_a_smem    : 1; // 0=register, 1=shared memory (B is always smem)
  uint32_t cd_nregs     : 2; // 0=8, 1=16, 2=32 C/D registers
  uint32_t fmt_s        : 5;
  uint32_t fmt_d        : 5;
  uint32_t step_m       : 4;
  uint32_t step_n       : 4;
  uint32_t step_k       : 4;
  uint32_t is_first_uop : 1; // set per-uop by tcu_uops expansion (C4)
  uint32_t is_last_uop  : 1;
  uint32_t is_setup_uop : 1; // WGMMA descriptor setup uop, no FEDP compute
};

// Helper: is_sparse derived from op_type (no per-uop bit).
inline bool tcu_is_sparse(TcuType t) {
  return t == TcuType::WMMA_SP || t == TcuType::WGMMA_SP;
}

// Helper: WGMMA family (covers dense + sparse).
inline bool tcu_is_wgmma(TcuType t) {
  return t == TcuType::WGMMA || t == TcuType::WGMMA_SP;
}

// Helper: WMMA family (covers dense + sparse).
inline bool tcu_is_wmma(TcuType t) {
  return t == TcuType::WMMA || t == TcuType::WMMA_SP;
}

inline std::ostream &operator<<(std::ostream &os, const TcuType& type) {
  switch (type) {
  case TcuType::WMMA:       os << "WMMA"; break;
  case TcuType::WGMMA:      os << "WGMMA"; break;
  case TcuType::WMMA_SP:    os << "WMMA.SP"; break;
  case TcuType::WGMMA_SP:   os << "WGMMA.SP"; break;
  case TcuType::TCU_LD:     os << "TCU_LD"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

using OpType = std::variant<
  AluType
, BrType
, MdvType
, LsuType
, AmoType
, FpuType
, CsrType
, VoteType
, ShflType
, WgatherType
, WctlType
#ifdef VX_CFG_EXT_DXA_ENABLE
, DxaType
#endif
#ifdef VX_CFG_EXT_TCU_ENABLE
, TcuType
#endif
#ifdef VX_CFG_EXT_TFU_ENABLE
, TfuType
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
, TexType
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
, OmType
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
, GfxwType
#endif
>;

using IntrArgs = std::variant<
  IntrAluArgs
, IntrBrArgs
, IntrMdvArgs
, IntrLsuArgs
, IntrAmoArgs
, IntrFpuArgs
, IntrCsrArgs
, IntrWgatherArgs
, IntrWctlArgs
#ifdef VX_CFG_EXT_DXA_ENABLE
, IntrDxaArgs
#endif
#ifdef VX_CFG_EXT_TCU_ENABLE
, IntrTcuArgs
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
, IntrTexArgs
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
, IntrOmArgs
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
, IntrGfxwArgs
#endif
>;

///////////////////////////////////////////////////////////////////////////////

enum class AddrType {
  Global,
  Shared,
  IO
};

inline AddrType get_addr_type(uint64_t addr) {
  if (addr >= VX_MEM_IO_BASE_ADDR && addr < VX_MEM_IO_END_ADDR) {
     return AddrType::IO;
  }
  if (VX_CFG_LMEM_ENABLED) {
    if (addr >= VX_MEM_LMEM_BASE_ADDR && (addr-VX_MEM_LMEM_BASE_ADDR) < uint64_t(VX_CFG_LMEM_SIZE)) {
        return AddrType::Shared;
    }
  }
  return AddrType::Global;
}

inline std::ostream &operator<<(std::ostream &os, const AddrType& type) {
  switch (type) {
  case AddrType::Global: os << "Global"; break;
  case AddrType::Shared: os << "Shared"; break;
  case AddrType::IO:     os << "IO"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

inline std::ostream &operator<<(std::ostream &os, const ArbiterType& type) {
  switch (type) {
  case ArbiterType::Priority:   os << "Priority"; break;
  case ArbiterType::RoundRobin: os << "RoundRobin"; break;
  case ArbiterType::Matrix:     os << "Matrix"; break;
  case ArbiterType::GreedyRR:   os << "GreedyRR"; break;
  case ArbiterType::GreedyMatrix: os << "GreedyMatrix"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

// Lanes of an LSU request or response: the core's LSU lanes or the D-cache
// channels behind them, whichever is wider.
inline constexpr uint32_t LSU_MAX_LANES = (LSU_CHANNELS > DCACHE_CHANNELS) ? LSU_CHANNELS : DCACHE_CHANNELS;

struct LsuReq {
  MemOp op;
  MemFlags flags;
  FixedVector<uint64_t, LSU_MAX_LANES> addrs;
  FixedVector<std::shared_ptr<mem_block_t>, LSU_MAX_LANES> data;
  FixedVector<uint64_t, LSU_MAX_LANES> byteen;
  // Comparand per lane. Only compare-and-swap reads it; it is a separate
  // field because the data operand is already carrying the swap value.
  FixedVector<uint64_t, LSU_MAX_LANES> amo_cmp;
  BitVector<> mask;
  FixedVector<uint32_t, LSU_MAX_LANES> tids;
  uint32_t tag;
  uint32_t cid;
  uint32_t wid;
  uint64_t uuid;

  LsuReq(uint32_t size)
    : op(MemOp::LD)
    , flags()
    , addrs(size, 0)
    , data(size)
    , byteen(size, 0)
    , amo_cmp(size, 0)
    , mask(size)
    , tids(size, 0)
    , tag(0)
    , cid(0)
    , wid(0)
    , uuid(0)
  {}

  bool is_write() const { return memop_is_write(op); }
  bool is_amo() const { return memop_is_atomic(op); }

  friend std::ostream &operator<<(std::ostream &os, const LsuReq& req) {
    os << "op=" << req.op << ", mask=" << req.mask << ", addr={";
    bool first_addr = true;
    for (size_t i = 0; i < req.mask.size(); ++i) {
      if (!first_addr) os << ", ";
      first_addr = false;
      if (req.mask.test(i)) {
        os << "0x" << std::hex << req.addrs.at(i) << std::dec;
      } else {
        os << "-";
      }
    }
    os << "}, tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct LsuRsp {
  BitVector<> mask;
  uint64_t tag;
  uint32_t cid;
  uint64_t uuid;

  // TLM data: per-lane block populated for loads/AMO returns.
  FixedVector<std::shared_ptr<mem_block_t>, LSU_MAX_LANES> data;

 LsuRsp(uint32_t size)
    : mask(size)
    , tag (0)
    , cid(0)
    , uuid(0)
    , data(size)
  {}

  friend std::ostream &operator<<(std::ostream &os, const LsuRsp& rsp) {
    os << "mask=" << rsp.mask << ", tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

// Maps RVA AmoType → MemOp. Signed/unsigned MIN/MAX variants collapse onto
// MemOp::AMO_MIN/MAX; the unsigned distinction rides on flags.amo_unsigned
// (set per-AMO by the LSU via amo_is_unsigned() below).
inline MemOp amo_to_memop(AmoType t) {
  switch (t) {
  case AmoType::LR:      return MemOp::AMO_LR;
  case AmoType::SC:      return MemOp::AMO_SC;
  case AmoType::AMOADD:  return MemOp::AMO_ADD;
  case AmoType::AMOSWAP: return MemOp::AMO_SWAP;
  case AmoType::AMOXOR:  return MemOp::AMO_XOR;
  case AmoType::AMOOR:   return MemOp::AMO_OR;
  case AmoType::AMOAND:  return MemOp::AMO_AND;
  case AmoType::AMOMIN:
  case AmoType::AMOMINU: return MemOp::AMO_MIN;
  case AmoType::AMOMAX:
  case AmoType::AMOMAXU: return MemOp::AMO_MAX;
  case AmoType::AMOCAS:  return MemOp::AMO_CAS;
  }
  std::abort();
}

// True for RVA MIN/MAX unsigned variants; carried as flags.amo_unsigned.
inline bool amo_is_unsigned(AmoType t) {
  return t == AmoType::AMOMINU || t == AmoType::AMOMAXU;
}

// Globally-unique per-hart id. Lower bits carry the most entropy
// (thread, then warp, then core) so the small per-bank reservation
// table sees fewer hash collisions.
inline uint32_t make_hart_id(uint32_t cid, uint32_t wid, uint32_t tid) {
  constexpr uint32_t LOG_WARPS   = log2ceil(VX_CFG_NUM_WARPS);
  constexpr uint32_t LOG_THREADS = log2ceil(VX_CFG_NUM_THREADS);
  return (cid << (LOG_WARPS + LOG_THREADS))
       | (wid << LOG_THREADS)
       | tid;
}

struct MemReq {
  MemOp    op;
  uint64_t addr;
  std::shared_ptr<mem_block_t> data;
  uint64_t byteen = 0;
  // Comparand, read only by compare-and-swap. The data block carries the
  // swap value, so the comparand needs a field of its own.
  uint64_t amo_cmp = 0;
  uint32_t tag;
  uint32_t hart_id;
  uint64_t uuid;
  MemFlags flags;
  // Under cache line merging: the tags of the ports merged into this read, by
  // offset in the leader's merge group. They ride the bank access and its
  // response, as in the hardware's bank tag.
  std::shared_ptr<const std::vector<uint32_t>> merged_tags;

  MemReq(MemOp _op = MemOp::LD,
         uint64_t _addr = 0,
         const std::shared_ptr<mem_block_t>& _data = nullptr,
         uint64_t _byteen = 0,
         uint64_t _tag = 0,
         uint32_t _hart_id = 0,
         uint64_t _uuid = 0
  ) : op(_op)
    , addr(_addr)
    , data(_data)
    , byteen(_byteen)
    , tag(_tag)
    , hart_id(_hart_id)
    , uuid(_uuid)
  {}

  bool is_write() const { return memop_is_write(op); }
  AddrType addr_type() const { return get_addr_type(addr); }

  friend std::ostream &operator<<(std::ostream &os, const MemReq& req) {
    os << "op=" << req.op
       << ", addr=0x" << std::hex << req.addr << std::dec
       << ", type=" << req.addr_type()
       << ", tag=0x" << std::hex << req.tag << std::dec
       << ", hart_id=" << req.hart_id
       << ", flags=" << req.flags
       << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct MemRsp {
  uint64_t tag;
  uint32_t hart_id;
  uint64_t uuid;
  std::shared_ptr<mem_block_t> data;
  std::shared_ptr<const std::vector<uint32_t>> merged_tags; // see MemReq

  MemRsp(uint64_t _tag = 0,
         uint32_t _hart_id = 0,
         uint64_t _uuid = 0,
         const std::shared_ptr<mem_block_t>& _data = nullptr
  ) : tag(_tag)
    , hart_id(_hart_id)
    , uuid(_uuid)
    , data(_data)
  {}

  friend std::ostream &operator<<(std::ostream &os, const MemRsp& rsp) {
    os << "tag=0x" << std::hex << rsp.tag << std::dec << ", hart_id=" << rsp.hart_id;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class HashTable {
public:
  typedef T DataType;

  HashTable(uint32_t capacity)
    : entries_(capacity)
    , size_(0)
  {}

  bool empty() const {
    return (0 == size_);
  }

  bool full() const {
    return (size_ == entries_.size());
  }

  uint32_t size() const {
    return size_;
  }

  bool contains(uint32_t index) const {
    return entries_.at(index).first;
  }

  const T& at(uint32_t index) const {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  T& at(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  uint32_t allocate(const T& value) {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      if (!entry.first) {
        entry.first = true;
        entry.second = value;
        ++size_;
        return i;
      }
    }
    assert(false);
    return -1;
  }

  void release(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    entry.first = false;
    --size_;
  }

  void clear() {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      entry.first = false;
    }
    size_ = 0;
  }

private:
  std::vector<std::pair<bool, T>> entries_;
  uint32_t size_;
};

///////////////////////////////////////////////////////////////////////////////

// Global-barrier control messages, carried on core <-> cluster event links.
struct GbarArrive {
  uint32_t bar_id;
  uint32_t count;
  uint32_t core_id;
};

struct GbarResume {
  uint32_t bar_id;
};

// A resolved warp-control op, registered from the SFU to the scheduler: the
// warp is released, or deactivated when it disabled all its threads.
struct WarpCtl {
  uint32_t wid;
  bool     exit;
};

// Fragment-work-distributor control messages, carried on raster-core <-> core
// event links.
struct FwdArm {
  Word frag_entry;
  Word frag_param;
};

struct FwdDone {
  uint32_t core_id;
};

///////////////////////////////////////////////////////////////////////////////

using LsuArbiter  = TxRxArbiter<LsuReq, LsuRsp>;
using MemArbiter  = TxRxArbiter<MemReq, MemRsp>;
using MemCrossBar = TxRxCrossBar<MemReq, MemRsp>;
using MemReqSlice = RegSlice<MemReq>;
using MemRspSlice = RegSlice<MemRsp>;

}
