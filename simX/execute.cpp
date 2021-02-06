#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "util.h"
#include "warp.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

struct DivergentBranchException {};

static bool checkUnanimous(unsigned p, 
                           const std::vector<std::vector<Reg<Word>>> &m,
                           const std::vector<bool> &tm) {
  bool same;
  unsigned i;
  for (i = 0; i < m.size(); ++i) {
    if (tm[i]) {
      same = m[i][p];
      break;
    }
  }
  if (i == m.size())
    throw DivergentBranchException();

  //std::cout << "same: " << same << "  with -> ";
  for (; i < m.size(); ++i) {
    if (tm[i]) {
      //std::cout << " " << (bool(m[i][p]));
      if (same != (bool(m[i][p]))) {
        //std::cout << " FALSE\n";
        return false;
      }
    }
  }
  //std::cout << " TRUE\n";
  return true;
}

void Warp::execute(Instr &instr, trace_inst_t *trace_inst) {
  /* If I try to execute a privileged instruction in user mode, throw an
     exception 3. */
  if (instr.getPrivileged() && !supervisorMode_) {
    D(3, "INTERRUPT SUPERVISOR\n");
    this->interrupt(3);
    return;
  }

  Size nextActiveThreads = activeThreads_;
  Size wordSz = core_->arch().getWordSize();
  Word nextPc = pc_;

  memAccesses_.clear();

  bool sjOnce(true);  // Has not yet split or joined once.
  bool pcSet(false);  // PC has already been set
  
  Word func3 = instr.getFunc3();
  Word func6 = instr.getFunc6();
  Word func7 = instr.getFunc7();

  Opcode opcode = instr.getOpcode();
  RegNum rdest  = instr.getRDest();
  RegNum rsrc0  = instr.getRSrc(0);
  RegNum rsrc1  = instr.getRSrc(1);
  RegNum pred   = instr.getPred();
  Word immsrc   = instr.getImm();
  bool vmask    = instr.getVmask();

  for (Size t = 0; t < activeThreads_; t++) {
    std::vector<Reg<Word>> &reg = regFile_[t];

    bool is_gpgpu = (opcode == GPGPU);
    bool is_tmc = is_gpgpu && (func3 == 0);
    bool is_wspawn = is_gpgpu && (func3 == 1);
    bool is_barrier = is_gpgpu && (func3 == 4);

    bool not_active = !tmask_[t];
    bool gpgpu_zero = (is_tmc || is_barrier || is_wspawn) && (t != 0);    

    if (not_active || gpgpu_zero)
      continue;

    ++insts_;

    switch (opcode) {
    case NOP:
      //std::cout << "NOP_INST\n";
      break;
    case R_INST: {
      // std::cout << "R_INST\n";
      Word m_exten = func7 & 0x1;
      if (m_exten) {
        // std::cout << "FOUND A MUL/DIV\n";

        switch (func3) {
        case 0:
          // MUL
          D(3, "MUL: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          reg[rdest] = ((int)reg[rsrc0]) * ((int)reg[rsrc1]);
          break;
        case 1:
          // MULH
          D(3, "MULH: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            int64_t first = (int64_t)reg[rsrc0];
            if (reg[rsrc0] & 0x80000000) {
              first = first | 0xFFFFFFFF00000000;
            }
            int64_t second = (int64_t)reg[rsrc1];
            if (reg[rsrc1] & 0x80000000) {
              second = second | 0xFFFFFFFF00000000;
            }
            // cout << "mulh: " << std::dec << first << " * " << second;
            uint64_t result = first * second;
            reg[rdest] = (result >> 32) & 0xFFFFFFFF;
            // cout << " = " << result << "   or  " <<  reg[rdest] << "\n";
          }
          break;
        case 2:
          // MULHSU
          D(3, "MULHSU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            int64_t first = (int64_t)reg[rsrc0];
            if (reg[rsrc0] & 0x80000000) {
              first = first | 0xFFFFFFFF00000000;
            }
            int64_t second = (int64_t)reg[rsrc1];
            reg[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 3:
          // MULHU
          D(3, "MULHU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            uint64_t first = (uint64_t)reg[rsrc0];
            uint64_t second = (uint64_t)reg[rsrc1];
            // cout << "MULHU\n";
            reg[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 4:
          // DIV
          D(3, "DIV: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (reg[rsrc1] == 0) {
            reg[rdest] = -1;
            break;
          }
          // cout << "dividing: " << std::dec << ((int) reg[rsrc0]) << " / " << ((int) reg[rsrc1]);
          reg[rdest] = ((int)reg[rsrc0]) / ((int)reg[rsrc1]);
          // cout << " = " << ((int) reg[rdest]) << "\n";
          break;
        case 5:
          // DIVU
          D(3, "DIVU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (reg[rsrc1] == 0) {
            reg[rdest] = -1;
            break;
          }
          reg[rdest] = ((uint32_t)reg[rsrc0]) / ((uint32_t)reg[rsrc1]);
          break;
        case 6:
          // REM
          D(3, "REM: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (reg[rsrc1] == 0) {
            reg[rdest] = reg[rsrc0];
            break;
          }
          reg[rdest] = ((int)reg[rsrc0]) % ((int)reg[rsrc1]);
          break;
        case 7:
          // REMU
          D(3, "REMU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (reg[rsrc1] == 0) {
            reg[rdest] = reg[rsrc0];
            break;
          }
          reg[rdest] = ((uint32_t)reg[rsrc0]) % ((uint32_t)reg[rsrc1]);
          break;
        default:
          std::cout << "unsupported MUL/DIV instr\n";
          std::abort();
        }
      } else {
        // std::cout << "NORMAL R-TYPE\n";
        switch (func3) {
        case 0:
          if (func7) {
            D(3, "SUBI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            reg[rdest] = reg[rsrc0] - reg[rsrc1];
            reg[rdest].trunc(wordSz);
          } else {
            D(3, "ADDI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            reg[rdest] = reg[rsrc0] + reg[rsrc1];
            reg[rdest].trunc(wordSz);
          }
          break;
        case 1:
          D(3, "SLLI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          reg[rdest] = reg[rsrc0] << reg[rsrc1];
          reg[rdest].trunc(wordSz);
          break;
        case 2:
          D(3, "SLTI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (int(reg[rsrc0]) < int(reg[rsrc1])) {
            reg[rdest] = 1;
          } else {
            reg[rdest] = 0;
          }
          break;
        case 3:
          D(3, "SLTU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (Word_u(reg[rsrc0]) < Word_u(reg[rsrc1])) {
            reg[rdest] = 1;
          } else {
            reg[rdest] = 0;
          }
          break;
        case 4:
          D(3, "XORI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          reg[rdest] = reg[rsrc0] ^ reg[rsrc1];
          break;
        case 5:
          if (func7) {
            D(3, "SRLI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            reg[rdest] = int(reg[rsrc0]) >> int(reg[rsrc1]);
            reg[rdest].trunc(wordSz);
          } else {
            D(3, "SRLU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            reg[rdest] = Word_u(reg[rsrc0]) >> Word_u(reg[rsrc1]);
            reg[rdest].trunc(wordSz);
          }
          break;
        case 6:
          D(3, "ORI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          reg[rdest] = reg[rsrc0] | reg[rsrc1];
          break;
        case 7:
          D(3, "ANDI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          reg[rdest] = reg[rsrc0] & reg[rsrc1];
          break;
        default:
          std::cout << "ERROR: UNSUPPORTED R INST\n";
          std::abort();
        }
      }
    } break;
    case L_INST: {
      Word memAddr = ((reg[rsrc0] + immsrc) & 0xFFFFFFFC);
      Word shift_by = ((reg[rsrc0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->mem().read(memAddr, supervisorMode_);
      trace_inst->is_lw = true;
      trace_inst->mem_addresses[t] = memAddr;
      switch (func3) {
      case 0:
        // LBI
        D(3, "LBI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LWI
        D(3, "LWI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LDI
        D(3, "LDI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = int(data_read & 0xFFFFFFFF);
        break;
      case 4:
        // LBU
        D(3, "LBU: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = unsigned((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LWU
        D(3, "LWU: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = unsigned((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();
        memAccesses_.push_back(Warp::MemAccess(false, memAddr));
      }
      D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
      D(3, "LOAD MEM DATA: " << std::hex << data_read);
    } break;
    case I_INST:
      //std::cout << "I_INST\n";
      switch (func3) {
      case 0:
        // ADDI
        D(3, "ADDI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        reg[rdest] = reg[rsrc0] + immsrc;
        reg[rdest].trunc(wordSz);
        break;
      case 2:
        // SLTI
        D(3, "SLTI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        if (int(reg[rsrc0]) < int(immsrc)) {
          reg[rdest] = 1;
        } else {
          reg[rdest] = 0;
        }
        break;
      case 3: {
        // SLTIU
        D(3, "SLTIU: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        if (unsigned(reg[rsrc0]) < unsigned(immsrc)) {
          reg[rdest] = 1;
        } else {
          reg[rdest] = 0;
        }
      } break;
      case 4:
        // XORI
        D(3, "XORI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = reg[rsrc0] ^ immsrc;
        break;
      case 6:
        // ORI
        D(3, "ORI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = reg[rsrc0] | immsrc;
        break;
      case 7:
        // ANDI
        D(3, "ANDI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = reg[rsrc0] & immsrc;
        break;
      case 1:
        // SLLI
        D(3, "SLLI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        reg[rdest] = reg[rsrc0] << immsrc;
        reg[rdest].trunc(wordSz);
        break;
      case 5:
        if ((func7 == 0)) {
          // SRLI
          D(3, "SRLI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
          Word result = Word_u(reg[rsrc0]) >> Word_u(immsrc);
          reg[rdest] = result;
          reg[rdest].trunc(wordSz);
        } else {
          // SRAI
          D(3, "SRAI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
          Word op1 = reg[rsrc0];
          Word op2 = immsrc;
          reg[rdest] = op1 >> op2;
          reg[rdest].trunc(wordSz);
        }
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();
      }
      break;
    case S_INST: {
      ++stores_;
      Word memAddr = reg[rsrc0] + immsrc;
      trace_inst->is_sw = true;
      trace_inst->mem_addresses[t] = memAddr;
      // //std::cout << "FUNC3: " << func3 << "\n";
      if ((memAddr == 0x00010000) && (t == 0)) {
        Word num = reg[rsrc1];
        fprintf(stderr, "%c", (char)num);
        break;
      }
      switch (func3) {
      case 0:
        // SB
        D(3, "SB: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, reg[rsrc1] & 0x000000FF, supervisorMode_, 1);
        break;
      case 1:
        // SH
        D(3, "SH: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, reg[rsrc1], supervisorMode_, 2);
        break;
      case 2:
        // SD
        D(3, "SD: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, reg[rsrc1], supervisorMode_, 4);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED S INST\n";
        std::abort();
      }
      D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
      memAccesses_.push_back(Warp::MemAccess(true, memAddr));
    } break;
    case B_INST:
      trace_inst->stall_warp = true;
      switch (func3) {
      case 0:
        // BEQ
        D(3, "BEQ: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(reg[rsrc0]) == int(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 1:
        // BNE
        D(3, "BNE: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(reg[rsrc0]) != int(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 4:
        // BLT
        D(3, "BLT: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(reg[rsrc0]) < int(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 5:
        // BGE
        D(3, "BGE: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(reg[rsrc0]) >= int(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 6:
        // BLTU
        D(3, "BLTU: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (Word_u(reg[rsrc0]) < Word_u(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 7:
        // BGEU
        D(3, "BGEU: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (Word_u(reg[rsrc0]) >= Word_u(reg[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      }
      break;
    case LUI_INST:
      D(3, "LUI: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      reg[rdest] = (immsrc << 12) & 0xfffff000;
      break;
    case AUIPC_INST:
      D(3, "AUIPC: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      reg[rdest] = ((immsrc << 12) & 0xfffff000) + (pc_ - 4);
      break;
    case JAL_INST:
      D(3, "JAL: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!pcSet) {
        nextPc = (pc_ - 4) + immsrc;
        //std::cout << "JAL... SETTING PC: " << nextPc << "\n";      
      }
      if (rdest != 0) {
        reg[rdest] = pc_;
      }
      pcSet = true;
      break;
    case JALR_INST:
      D(3, "JALR: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!pcSet) {
        nextPc = reg[rsrc0] + immsrc;
        //std::cout << "JALR... SETTING PC: " << nextPc << "\n";
      }
      if (rdest != 0) {
        reg[rdest] = pc_;
      }
      pcSet = true;
      break;
    case SYS_INST: {
      D(3, "SYS_INST: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
      Word rs1 = reg[rsrc0];
      Word csr_addr = immsrc & 0x00000FFF;
      // GPGPU CSR extension
      if (csr_addr == CSR_WTID) {
        // Warp threadID
        reg[rdest] = t;
      } else if (csr_addr == CSR_LTID) {
        // Core threadID
        reg[rdest] = t + 
                     id_ * core_->arch().getNumThreads();
      } else if (csr_addr == CSR_GTID) {
        // Processor threadID
        reg[rdest] = t + 
                     id_ * core_->arch().getNumThreads() + 
                     core_->arch().getNumThreads() * core_->arch().getNumWarps() * core_->id();
      } else if (csr_addr == CSR_LWID) {
        // Core warpID
        reg[rdest] = id_;
      } else if (csr_addr == CSR_GWID) {
        // Processor warpID        
        reg[rdest] = id_ + core_->arch().getNumWarps() * core_->id();
      } else if (csr_addr == CSR_GCID) {
        // Processor coreID
        reg[rdest] = core_->id();
      } else if (csr_addr == CSR_NT) {
        // Number of threads per warp
        reg[rdest] = core_->arch().getNumThreads();
      } else if (csr_addr == CSR_NW) {
        // Number of warps per core
        reg[rdest] = core_->arch().getNumWarps();
      } else if (csr_addr == CSR_NC) {
        // Number of cores
        reg[rdest] = core_->arch().getNumCores();
      } else if (csr_addr == CSR_INSTRET) {
        // NumInsts
        reg[rdest] = (Word)core_->num_instructions();
      } else if (csr_addr == CSR_INSTRET_H) {
        // NumInsts
        reg[rdest] = (Word)(core_->num_instructions() >> 32);
      } else if (csr_addr == CSR_CYCLE) {
        // NumCycles
        reg[rdest] = (Word)core_->num_steps();
      } else if (csr_addr == CSR_CYCLE_H) {
        // NumCycles
        reg[rdest] = (Word)(core_->num_steps() >> 32);
      } else {
        switch (func3) {
        case 0:
          if (csr_addr < 2) {
            // ECALL/EBREAK
            nextActiveThreads = 0;
            spawned_ = false;
          }
          break;
        case 1:
          // CSRRW
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1;
          break;
        case 2:
          // CSRRS
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1 | csrs_[csr_addr];
          break;
        case 3:
          // CSRRC
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1 & (~csrs_[csr_addr]);
          break;
        case 5:
          // CSRRWI
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0;
          break;
        case 6:
          // CSRRSI
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0 | csrs_[csr_addr];
          break;
        case 7:
          // CSRRCI
          if (rdest != 0) {
            reg[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0 & (~csrs_[csr_addr]);
          break;
        default:
          break;
        }
      }
    } break;
    case FENCE:
      D(3, "FENCE");
      break;
    case PJ_INST:
      D(3, "PJ_INST: r" << rsrc0 << ", r" << rsrc1);
      if (reg[rsrc0]) {
        if (!pcSet)
          nextPc = reg[rsrc1];
        pcSet = true;
      }
      break;
    case GPGPU:
      switch (func3) {
      case 1:
        // WSPAWN
        D(3, "WSPAWN: r" << rsrc0 << ", r" << rsrc1);
        trace_inst->wspawn = true;
        if (sjOnce) {
          sjOnce = false;
          unsigned num_to_wspawn = std::min<unsigned>(reg[rsrc0], core_->arch().getNumWarps());
          D(0, "Spawning " << num_to_wspawn << " new warps at PC: " << std::hex << reg[rsrc1]);
          for (unsigned i = 1; i < num_to_wspawn; ++i) {
            Warp &newWarp(core_->warp(i));
            {
              newWarp.set_pc(reg[rsrc1]);
              for (size_t kk = 0; kk < tmask_.size(); kk++) {
                if (kk == 0) {
                  newWarp.setTmask(kk, true);
                } else {
                  newWarp.setTmask(kk, false);
                }
              }
              newWarp.setActiveThreads(1);
              newWarp.setSupervisorMode(false);
              newWarp.setSpawned(true);
            }
          }
          break;
        }
        break;
      case 2: {
        // SPLIT
        D(3, "SPLIT: r" << pred);
        trace_inst->stall_warp = true;
        if (sjOnce) {
          sjOnce = false;
          if (checkUnanimous(pred, regFile_, tmask_)) {
            D(3, "Unanimous pred: " << pred << "  val: " << reg[pred] << "\n");
            DomStackEntry e(tmask_);
            e.uni = true;
            domStack_.push(e);
            break;
          }
          D(3, "Split: Original TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )

          DomStackEntry e(pred, regFile_, tmask_, pc_);
          domStack_.push(tmask_);
          domStack_.push(e);
          for (unsigned i = 0; i < e.tmask.size(); ++i) {
            tmask_[i] = !e.tmask[i] && tmask_[i];
          }

          D(3, "Split: New TM");
          DX( for (auto y : tmask_) D(3, y << " "); )
          D(3, "Split: Pushed TM PC: " << std::hex << e.pc << std::dec << "\n");
          DX( for (auto y : e.tmask) D(3, y << " "); )
        }
        break;
      }
      case 3:
        // JOIN
        D(3, "JOIN");
        if (sjOnce) {
          sjOnce = false;
          if (!domStack_.empty() && domStack_.top().uni) {
            D(2, "Uni branch at join");
            printf("NEW DOMESTACK: \n");
            tmask_ = domStack_.top().tmask;
            domStack_.pop();
            break;
          }
          if (!domStack_.top().fallThrough) {
            if (!pcSet) {
              nextPc = domStack_.top().pc;
              D(3, "join: NOT FALLTHROUGH PC: " << std::hex << nextPc << std::dec);
            }
            pcSet = true;
          }

          D(3, "Join: Old TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )
          std::cout << "\n";
          tmask_ = domStack_.top().tmask;

          D(3, "Join: New TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )

          domStack_.pop();
        }
        break;
      case 4:
        trace_inst->stall_warp = true;
        // is_barrier
        break;
      case 0:
        // TMC
        D(3, "TMC: r" << rsrc0);
        trace_inst->stall_warp = true;
        nextActiveThreads = std::min<unsigned>(reg[rsrc0], core_->arch().getNumThreads());
        {
          for (size_t ff = 0; ff < tmask_.size(); ff++) {
            if (ff < nextActiveThreads) {
              tmask_[ff] = true;
            } else {
              tmask_[ff] = false;
            }
          }
        }
        if (nextActiveThreads == 0) {
          spawned_ = false;
        }
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED GPGPU INSTRUCTION " << instr << "\n";
      }
      break;
    case VSET_ARITH: {
      D(3, "VSET_ARITH");
      int VLMAX = (instr.getVlmul() * VLEN_) / instr.getVsew();
      switch (func3) {
      case 0: // vector-vector
        trace_inst->vs1 = rsrc0;
        trace_inst->vs2 = rsrc1;
        trace_inst->vd  = rdest;
        switch (func6) {
        case 0: {
          D(3, "Addition " << rsrc0 << " " << rsrc1 << " Dest:" << rdest);
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          std::vector<Reg<char *>> &mask = vregFile_[0];

          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *mask_ptr = (uint8_t *)mask[i].value();
              uint8_t value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                uint8_t *first_ptr = (uint8_t *)vr1[i].value();
                uint8_t *second_ptr = (uint8_t *)vr2[i].value();
                uint8_t result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                uint8_t *result_ptr = (uint8_t *)vd[i].value();
                *result_ptr = result;
              }
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *mask_ptr = (uint16_t *)mask[i].value();
              uint16_t value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                uint16_t *first_ptr = (uint16_t *)vr1[i].value();
                uint16_t *second_ptr = (uint16_t *)vr2[i].value();
                uint16_t result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                uint16_t *result_ptr = (uint16_t *)vd[i].value();
                *result_ptr = result;
              }
            }
          } else if (vtype_.vsew == 32) {
            D(3, "Doing 32 bit vector addition");
            for (int i = 0; i < vl_; i++) {
              int *mask_ptr = (int *)mask[i].value();
              int value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                int *first_ptr = (int *)vr1[i].value();
                int *second_ptr = (int *)vr2[i].value();
                int result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                int *result_ptr = (int *)vd[i].value();
                *result_ptr = result;
              }
            }
          }
          
          DX( 
            D(3, "Vector Register state after addition:");
            for (size_t i = 0; i < vregFile_.size(); i++) {
              for (size_t j = 0; j < vregFile_[0].size(); j++) {
                if (vtype_.vsew == 8) {
                  uint8_t *ptr_val = (uint8_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                } else if (vtype_.vsew == 16) {
                  uint16_t *ptr_val = (uint16_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                } else if (vtype_.vsew == 32) {
                  uint32_t *ptr_val = (uint32_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                }
              }
            }
            D(3, "After vector register state after addition");
          )    
                
        } break;
        case 24: //vmseq
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 25: //vmsne
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 26: //vmsltu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 27: //vmslt
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 28: //vmsleu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 29: //vmsle
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 30: //vmsgtu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 31: //vmsgt
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        }
        break;
      case 2: {
        trace_inst->vs1 = rsrc0;
        trace_inst->vs2 = rsrc1;
        trace_inst->vd = rdest;

        switch (func6) {
        case 24: //vmandnot
        {
          D(3, "vmandnot");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 25: //vmand
        {
          D(3, "vmand");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 26: //vmor
        {
          D(3, "vmor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            D(3, "VLMAX: " << VLMAX);
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 27: //vmxor
        {
          D(3, "vmxor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            uint8_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 28: //vmornot
        {
          D(3, "vmornot");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 29: //vmnand
        {
          D(3, "vmnand");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 30: //vmnor
        {
          D(3, "vmnor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            uint8_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 31: //vmxnor
        {
          D(3, "vmxnor");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 37: //vmul
        {
          D(3, "vmul");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 45: //vmacc
        {
          D(3, "vmacc");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        }
      } break;
      case 6: {
        switch (func6) {
        case 0: {
          D(3, "vmadd.vx");
          uint8_t *result_ptr;

          //vector<Reg<char *>> & vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (reg[rsrc0] + *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              //uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (reg[rsrc0] + *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              //uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (reg[rsrc0] + *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 37: //vmul.vx
        {
          D(3, "vmul.vx");
          uint8_t *result_ptr;

          //vector<Reg<char *>> & vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (reg[rsrc0] * *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              //uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (reg[rsrc0] * *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              //uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (reg[rsrc0] * *second_ptr);
              D(3, "Comparing " << reg[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        }
      } break;
      case 7: {
        vtype_.vill = 0; //TODO
        vtype_.vediv = instr.getVediv();
        vtype_.vsew  = instr.getVsew();
        vtype_.vlmul = instr.getVlmul();

        D(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << reg[rsrc0] << "VLMAX" << VLMAX);

        int s0 = reg[rsrc0];

        if (s0 <= VLMAX) {
          vl_ = s0;
        } else if (s0 < (2 * VLMAX)) {
          vl_ = (int)ceil((s0 * 1.0) / 2.0);
          D(3, "Length:" << vl_ << ceil(s0 / 2));
        } else if (s0 >= (2 * VLMAX)) {
          vl_ = VLMAX;
        }
        
        reg[rdest] = vl_;
        D(3, "VL:" << reg[rdest]);

        Word regNum(0);

        vregFile_.clear();
        for (int j = 0; j < 32; j++) {
          vregFile_.push_back(std::vector<Reg<char *>>());
          for (Word i = 0; i < (VLEN_ / instr.getVsew()); ++i) {
            int *elem_ptr = (int *)malloc(instr.getVsew() / 8);
            for (Word f = 0; f < (instr.getVsew() / 32); f++)
              elem_ptr[f] = 0;
            vregFile_[j].push_back(Reg<char *>(id_, regNum++, (char *)elem_ptr));
          }
        }
      } break;
      default: {
        std::cout << "default???\n" << std::flush;
      }
      }
    } break;
    case VL: {
      D(3, "Executing vector load");
      
      D(3, "lmul: " << vtype_.vlmul << " VLEN:" << VLEN_ << "sew: " << vtype_.vsew);
      D(3, "src: " << rsrc0 << " " << reg[rsrc0]);
      D(3, "dest" << rdest);
      D(3, "width" << instr.getVlsWidth());

      std::vector<Reg<char *>> &vd = vregFile_[rdest];

      switch (instr.getVlsWidth()) {
      case 6: //load word and unit strided (not checking for unit stride)
      {
        for (int i = 0; i < vl_; i++) {
          Word memAddr = ((reg[rsrc0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
          Word data_read = core_->mem().read(memAddr, supervisorMode_);
          D(3, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
          int *result_ptr = (int *)vd[i].value();
          *result_ptr = data_read;

          trace_inst->is_lw = true;
          trace_inst->mem_addresses[i] = memAddr;
        }
        D(3, "Vector Register state ----:");
        // cout << "Finished loop" << std::endl;
      }
      // cout << "aaaaaaaaaaaaaaaaaaaaaa" << std::endl;
      break;
      default: {
        std::cout << "Serious default??\n" << std::flush;
      } break;
      }
      break;
    } break;
    case VS:
      for (int i = 0; i < vl_; i++) {
        // cout << "iter" << std::endl;
        ++stores_;
        Word memAddr = reg[rsrc0] + (i * vtype_.vsew / 8);
        // std::cout << "STORE MEM ADDRESS *** : " << std::hex << memAddr << "\n";

        trace_inst->is_sw = true;
        trace_inst->mem_addresses[i] = memAddr;

        switch (instr.getVlsWidth()) {
        case 6: //store word and unit strided (not checking for unit stride)
        {
          uint32_t *ptr_val = (uint32_t *)vregFile_[instr.getVs3()][i].value();
          D(3, "value: " << std::flush << (*ptr_val) << std::flush);
          core_->mem().write(memAddr, *ptr_val, supervisorMode_, 4);
          D(3, "store: " << memAddr << " value:" << *ptr_val << std::flush);
        } break;
        default:
          std::cout << "ERROR: UNSUPPORTED S INST\n" << std::flush;
          std::abort();
        }
        // cout << "Loop finished" << std::endl;
      }

      // cout << "After for loop" << std::endl;
      break;
    default:
      D(3, "pc: " << std::hex << (pc_ - 4));
      D(3, "ERROR: Unsupported instruction: " << instr);
      std::abort();
    }
  }

  activeThreads_ = nextActiveThreads;

  // This way, if pc was set by a side effect (such as interrupt), it will
  // retain its new value.
  if (pcSet) {
    pc_ = nextPc;
    D(3, "Next PC: " << std::hex << nextPc << std::dec);
  }

  if (nextActiveThreads > regFile_.size()) {
    std::cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
              << regFile_.size() << " available.\n";
    abort();
  }
}
