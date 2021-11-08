#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bitset>
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <util.h>
#include <rvfloats.h>
#include "warp.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

static bool HasDivergentThreads(const ThreadMask &thread_mask,                                
                                const std::vector<std::vector<Word>> &reg_file,
                                unsigned reg) {
  bool cond;
  size_t thread_idx = 0;
  size_t num_threads = reg_file.size();
  for (; thread_idx < num_threads; ++thread_idx) {
    if (thread_mask[thread_idx]) {
      cond = bool(reg_file[thread_idx][reg]);
      break;
    }
  }  
  assert(thread_idx != num_threads);  
  for (; thread_idx < num_threads; ++thread_idx) {
    if (thread_mask[thread_idx]) {
      if (cond != (bool(reg_file[thread_idx][reg]))) {
        return true;
      }
    }
  }
  return false;
}

inline uint32_t get_fpu_rm(uint32_t func3, Core* core, uint32_t tid, uint32_t wid) {
  return (func3 == 0x7) ? core->get_csr(CSR_FRM, tid, wid) : func3;
}

inline void update_fcrs(uint32_t fflags, Core* core, uint32_t tid, uint32_t wid) {
  if (fflags) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | fflags, tid, wid);
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | fflags, tid, wid);
  }
}

void Warp::execute(const Instr &instr, Pipeline *pipeline) {
  assert(tmask_.any());

  Word nextPC = PC_ + core_->arch().wsize();
  bool runOnce = false;
  
  Word func3 = instr.getFunc3();
  Word func6 = instr.getFunc6();
  Word func7 = instr.getFunc7();

  auto opcode = instr.getOpcode();
  int rdest  = instr.getRDest();
  int rsrc0  = instr.getRSrc(0);
  int rsrc1  = instr.getRSrc(1);
  Word immsrc= instr.getImm();
  Word vmask = instr.getVmask();

  int num_threads = core_->arch().num_threads();
  for (int t = 0; t < num_threads; t++) {
    if (!tmask_.test(t) || runOnce)
      continue;
    
    auto &iregs = iRegFile_.at(t);
    auto &fregs = fRegFile_.at(t);

    Word rsdata[3];
    Word rddata;

    int num_rsrcs = instr.getNRSrc();
    if (num_rsrcs) {    
      DPH(2, "[" << std::dec << t << "] Src Regs: ");
      for (int i = 0; i < num_rsrcs; ++i) {    
        int rst = instr.getRSType(i);
        int rs = instr.getRSrc(i);        
        if (i) DPN(2, ", ");
        switch (rst) {
        case 1: 
          rsdata[i] = iregs[rs];
          DPN(2, "r" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        case 2: 
          rsdata[i] = fregs[rs];
          DPN(2, "fr" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        default: break;
        }
      }
      DPN(2, std::endl);
    }

    bool rd_write = false;
  
    switch (opcode) {
    case NOP:
      break;
    case LUI_INST:
      rddata = (immsrc << 12) & 0xfffff000;
      rd_write = true;
      break;
    case AUIPC_INST:
      rddata = ((immsrc << 12) & 0xfffff000) + PC_;
      rd_write = true;
      break;
    case R_INST: {
      if (func7 & 0x1) {
        switch (func3) {
        case 0:
          // MUL
          rddata = ((WordI)rsdata[0]) * ((WordI)rsdata[1]);
          break;
        case 1: {
          // MULH
          int64_t first = (int64_t)rsdata[0];
          if (rsdata[0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)rsdata[1];
          if (rsdata[1] & 0x80000000) {
            second = second | 0xFFFFFFFF00000000;
          }
          uint64_t result = first * second;
          rddata = (result >> 32) & 0xFFFFFFFF;
        } break;
        case 2: {
          // MULHSU          
          int64_t first = (int64_t)rsdata[0];
          if (rsdata[0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)rsdata[1];
          rddata = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 3: {
          // MULHU
          uint64_t first = (uint64_t)rsdata[0];
          uint64_t second = (uint64_t)rsdata[1];
          rddata = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 4: {
          // DIV
          WordI dividen = rsdata[0];
          WordI divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            rddata = dividen;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 5: {
          // DIVU
          Word dividen = rsdata[0];
          Word divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 6: {
          // REM
          WordI dividen = rsdata[0];
          WordI divisor = rsdata[1];
          if (rsdata[1] == 0) {
            rddata = dividen;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            rddata = 0;
          } else {
            rddata = dividen % divisor;
          }
        } break;
        case 7: {
          // REMU
          Word dividen = rsdata[0];
          Word divisor = rsdata[1];
          if (rsdata[1] == 0) {
            rddata = dividen;
          } else {
            rddata = dividen % divisor;
          }
        } break;
        default:
          std::cout << "unsupported MUL/DIV instr\n";
          std::abort();
        }
      } else {
        switch (func3) {
        case 0:
          if (func7) {
            rddata = rsdata[0] - rsdata[1];
          } else {
            rddata = rsdata[0] + rsdata[1];
          }
          break;
        case 1:
          rddata = rsdata[0] << rsdata[1];
          break;
        case 2:
          rddata = (WordI(rsdata[0]) < WordI(rsdata[1]));
          break;
        case 3:
          rddata = (Word(rsdata[0]) < Word(rsdata[1]));
          break;
        case 4:
          rddata = rsdata[0] ^ rsdata[1];
          break;
        case 5:
          if (func7) {
            rddata = WordI(rsdata[0]) >> WordI(rsdata[1]);
          } else {
            rddata = Word(rsdata[0]) >> Word(rsdata[1]);
          }
          break;
        case 6:
          rddata = rsdata[0] | rsdata[1];
          break;
        case 7:
          rddata = rsdata[0] & rsdata[1];
          break;
        default:
          std::abort();
        }
      }
      rd_write = true;
    } break;
    case I_INST:
      switch (func3) {
      case 0:
        // ADDI
        rddata = rsdata[0] + immsrc;
        break;
      case 1:
        // SLLI
        rddata = rsdata[0] << immsrc;
        break;
      case 2:
        // SLTI
        rddata = (WordI(rsdata[0]) < WordI(immsrc));
        break;
      case 3: {
        // SLTIU
        rddata = (Word(rsdata[0]) < Word(immsrc));
      } break;
      case 4:
        // XORI
        rddata = rsdata[0] ^ immsrc;
        break;
      case 5:
        if (func7) {
          // SRAI
          Word result = WordI(rsdata[0]) >> immsrc;
          rddata = result;
        } else {
          // SRLI
          Word result = Word(rsdata[0]) >> immsrc;
          rddata = result;
        }
        break;
      case 6:
        // ORI
        rddata = rsdata[0] | immsrc;
        break;
      case 7:
        // ANDI
        rddata = rsdata[0] & immsrc;
        break;
      default:
        std::abort();
      }
      rd_write = true;
      break;
    case B_INST:
      switch (func3) {
      case 0:
        // BEQ
        if (rsdata[0] == rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 1:
        // BNE
        if (rsdata[0] != rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 4:
        // BLT
        if (WordI(rsdata[0]) < WordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 5:
        // BGE
        if (WordI(rsdata[0]) >= WordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 6:
        // BLTU
        if (Word(rsdata[0]) < Word(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 7:
        // BGEU
        if (Word(rsdata[0]) >= Word(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      }
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case JAL_INST:
      rddata = nextPC;
      nextPC = PC_ + immsrc;  
      pipeline->stall_warp = true;
      runOnce = true;
      rd_write = true;
      break;
    case JALR_INST:
      rddata = nextPC;
      nextPC = rsdata[0] + immsrc;
      pipeline->stall_warp = true;
      runOnce = true;
      rd_write = true;
      break;
    case L_INST: {
      Word memAddr   = ((rsdata[0] + immsrc) & 0xFFFFFFFC); // word aligned
      Word shift_by  = ((rsdata[0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->dcache_read(memAddr, 4);
      D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
      switch (func3) {
      case 0:
        // LBI
        rddata = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LHI
        rddata = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LW
        rddata = data_read;
        break;
      case 4:
        // LBU
        rddata = Word((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LHU
        rddata = Word((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::abort();        
      }
      rd_write = true;
    } break;
    case S_INST: {
      Word memAddr = rsdata[0] + immsrc;
      D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      switch (func3) {
      case 0:
        // SB
        core_->dcache_write(memAddr, rsdata[1] & 0x000000FF, 1);
        break;
      case 1:
        // SH
        core_->dcache_write(memAddr, rsdata[1], 2);
        break;
      case 2:
        // SW
        core_->dcache_write(memAddr, rsdata[1], 4);
        break;
      default:
        std::abort();
      }
    } break;
    case SYS_INST: {
      Word csr_addr = immsrc & 0x00000FFF;
      Word csr_value = core_->get_csr(csr_addr, t, id_);
      switch (func3) {
      case 0:
        if (csr_addr < 2) {
          // ECALL/EBREAK
          core_->trigger_ebreak();
        }
        break;
      case 1:
        // CSRRW
        rddata = csr_value;
        core_->set_csr(csr_addr, rsdata[0], t, id_);
        rd_write = true;
        break;
      case 2:
        // CSRRS
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsdata[0], t, id_);
        rd_write = true;
        break;
      case 3:
        // CSRRC
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsdata[0], t, id_);
        rd_write = true;
        break;
      case 5:
        // CSRRWI
        rddata = csr_value;
        core_->set_csr(csr_addr, rsrc0, t, id_);
        rd_write = true;
        break;
      case 6:
        // CSRRSI
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
        rd_write = true;
        break;
      case 7:
        // CSRRCI
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsrc0, t, id_);
        rd_write = true;
        break;
      default:
        break;
      }
    } break;
    case FENCE:
      pipeline->stall_warp = true; 
      runOnce = true;
      break;
    case (FL | VL):
      if (func3 == 0x2) {
        Word memAddr = rsdata[0] + immsrc;
        Word data_read = core_->dcache_read(memAddr, 4);        
        D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
        rddata = data_read;
      } else {  
        D(3, "Executing vector load");      
        D(3, "lmul: " << vtype_.vlmul << " VLEN:" << (core_->arch().vsize() * 8) << "sew: " << vtype_.vsew);
        D(3, "src: " << rsrc0 << " " << rsdata[0]);
        D(3, "dest" << rdest);
        D(3, "width" << instr.getVlsWidth());

        auto &vd = vRegFile_[rdest];

        switch (instr.getVlsWidth()) {
        case 6: { 
          //load word and unit strided (not checking for unit stride)
          for (int i = 0; i < vl_; i++) {
            Word memAddr = ((rsdata[0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
            D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
            Word data_read = core_->dcache_read(memAddr, 4);
            D(3, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
            int *result_ptr = (int *)(vd.data() + i);
            *result_ptr = data_read;            
          }
        } break;
        default:
          std::abort();
        }
        break;
      } 
      rd_write = true;
      break;
    case (FS | VS):
      if (func3 == 0x2) {
        Word memAddr = rsdata[0] + immsrc;
        core_->dcache_write(memAddr, rsdata[1], 4);
        D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      } else {
        for (int i = 0; i < vl_; i++) {
          Word memAddr = rsdata[0] + (i * vtype_.vsew / 8);
          D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
          switch (instr.getVlsWidth()) {
          case 6: {
            //store word and unit strided (not checking for unit stride)          
            uint32_t value = *(uint32_t *)(vRegFile_[instr.getVs3()].data() + i);
            core_->dcache_write(memAddr, value, 4);
            D(3, "store: " << memAddr << " value:" << value);
          } break;
          default:
            std::abort();
          }          
        }
      }
      break;    
    case FCI: { 
      uint32_t frm = get_fpu_rm(func3, core_, t, id_);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: //FADD
        rddata = rv_fadd(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x04: //FSUB
        rddata = rv_fsub(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x08: //FMUL
        rddata = rv_fmul(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x0c: //FDIV
        rddata = rv_fdiv(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x2c: //FSQRT
        rddata = rv_fsqrt(rsdata[0], frm, &fflags);
        break;        
      case 0x10:
        switch (func3) {            
        case 0: // FSGNJ.S
          rddata = rv_fsgnj(rsdata[0], rsdata[1]);
          break;          
        case 1: // FSGNJN.S
          rddata = rv_fsgnjn(rsdata[0], rsdata[1]);
          break;          
        case 2: // FSGNJX.S
          rddata = rv_fsgnjx(rsdata[0], rsdata[1]);
          break;
        }
        break;
      case 0x14:                
        if (func3) {
          // FMAX.S
          rddata = rv_fmax(rsdata[0], rsdata[1], &fflags);
        } else {
          // FMIN.S
          rddata = rv_fmin(rsdata[0], rsdata[1], &fflags);
        }
        break;
      case 0x60:
        if (rsrc1 == 0) { 
          // FCVT.W.S
          rddata = rv_ftoi(rsdata[0], frm, &fflags);
        } else {
          // FCVT.WU.S
          rddata = rv_ftou(rsdata[0], frm, &fflags);
        }
        break;
      case 0x70:      
        if (func3) {
          // FCLASS.S
          rddata = rv_fclss(rsdata[0]);
        } else {          
          // FMV.X.W
          rddata = rsdata[0];
        } 
        break;
      case 0x50:          
        switch(func3) {              
        case 0:
          // FLE.S
          rddata = rv_fle(rsdata[0], rsdata[1], &fflags);    
          break;              
        case 1:
          // FLT.S
          rddata = rv_flt(rsdata[0], rsdata[1], &fflags);
          break;              
        case 2:
          // FEQ.S
          rddata = rv_feq(rsdata[0], rsdata[1], &fflags);
          break;
        } break;        
      case 0x68:
        if (rsrc1) {
          // FCVT.S.WU:
          rddata = rv_utof(rsdata[0], frm, &fflags);
        } else {
          // FCVT.S.W:
          rddata = rv_itof(rsdata[0], frm, &fflags);
        }
        break;
      case 0x78:
        // FMV.W.X
        rddata = rsdata[0];
        break;
      }
      update_fcrs(fflags, core_, t, id_);
      rd_write = true;
    } break;
    case FMADD:      
    case FMSUB:      
    case FMNMADD:
    case FMNMSUB: {
      int frm = get_fpu_rm(func3, core_, t, id_);
      Word fflags = 0;
      switch (opcode) {
      case FMADD:
        rddata = rv_fmadd(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      case FMSUB:
        rddata = rv_fmsub(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      case FMNMADD:
        rddata = rv_fnmadd(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;  
      case FMNMSUB:
        rddata = rv_fnmsub(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      default:
        break;
      }              
      update_fcrs(fflags, core_, t, id_);
      rd_write = true;
    } break;
    case GPGPU:
      switch (func3) {
      case 0: {
        // TMC        
        if (rsrc1) {
          // predicate mode
          ThreadMask pred;
          for (int i = 0; i < num_threads; ++i) {
            pred[i] = tmask_[i] ? (iRegFile_[i][rsrc0] != 0) : 0;
          }
          if (pred.any()) {
            tmask_ &= pred;
          }
        } else {
          tmask_.reset();
          for (int i = 0; i < num_threads; ++i) {
            tmask_[i] = rsdata[0] & (1 << i);
          }
        }
        D(3, "*** TMC " << tmask_);
        active_ = tmask_.any();
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 1: {
        // WSPAWN
        int active_warps = std::min<int>(rsdata[0], core_->arch().num_warps());
        D(3, "*** Spawning " << (active_warps-1) << " warps at PC: " << std::hex << rsdata[1]);
        for (int i = 1; i < active_warps; ++i) {
          Warp &newWarp = core_->warp(i);
          newWarp.setPC(rsdata[1]);
          newWarp.setTmask(0, true);
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 2: {
        // SPLIT    
        if (HasDivergentThreads(tmask_, iRegFile_, rsrc0)) {          
          ThreadMask tmask;
          for (int i = 0; i < num_threads; ++i) {
            tmask[i] = tmask_[i] && !iRegFile_[i][rsrc0];
          }

          DomStackEntry e(tmask, nextPC);
          domStack_.push(tmask_);
          domStack_.push(e);
          for (size_t i = 0; i < e.tmask.size(); ++i) {
            tmask_[i] = !e.tmask[i] && tmask_[i];
          }
          active_ = tmask_.any();

          DPH(3, "*** Split: New TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, tmask_[num_threads-i-1]);
          DPN(3, ", Pushed TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, e.tmask[num_threads-i-1]);
          DPN(3, ", PC=0x" << std::hex << e.PC << "\n");
        } else {
          D(3, "*** Unanimous pred");
          DomStackEntry e(tmask_);
          e.unanimous = true;
          domStack_.push(e);
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 3: {
        // JOIN
        if (!domStack_.empty() && domStack_.top().unanimous) {
          D(3, "*** Uninimous branch at join");
          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();
          domStack_.pop();
        } else {
          if (!domStack_.top().fallThrough) {
            nextPC = domStack_.top().PC;
            D(3, "*** Join: next PC: " << std::hex << nextPC << std::dec);
          }

          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();

          DPH(3, "*** Join: New TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, tmask_[num_threads-i-1]);
          DPN(3, "\n");

          domStack_.pop();
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 4: {
        // BAR
        active_ = false;
        core_->barrier(rsdata[0], rsdata[1], id_);
        pipeline->stall_warp = true; 
        runOnce = true;       
      } break;
      case 6: {
        // PREFETCH
        int addr = rsdata[0];
        printf("*** PREFETCHED %d ***\n", addr);
      } break;
      default:
        std::abort();
      }
      break;
    case VSET: {
      int VLEN = core_->arch().vsize() * 8;
      int VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
      switch (func3) {
      case 0: // vector-vector
        switch (func6) {
        case 0: {
          auto& vr1 = vRegFile_[rsrc0];
          auto& vr2 = vRegFile_[rsrc1];
          auto& vd = vRegFile_[rdest];
          auto& mask = vRegFile_[0];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t emask = *(uint8_t *)(mask.data() + i);
              uint8_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint8_t first  = *(uint8_t *)(vr1.data() + i);
                uint8_t second = *(uint8_t *)(vr2.data() + i);
                uint8_t result = first + second;
                D(3, "Adding " << first << " + " << second << " = " << result);
                *(uint8_t *)(vd.data() + i) = result;
              }
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t emask = *(uint16_t *)(mask.data() + i);
              uint16_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint16_t first  = *(uint16_t *)(vr1.data() + i);
                uint16_t second = *(uint16_t *)(vr2.data() + i);
                uint16_t result = first + second;
                D(3, "Adding " << first << " + " << second << " = " << result);
                *(uint16_t *)(vd.data() + i) = result;
              }
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t emask = *(uint32_t *)(mask.data() + i);
              uint32_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint32_t first  = *(uint32_t *)(vr1.data() + i);
                uint32_t second = *(uint32_t *)(vr2.data() + i);
                uint32_t result = first + second;
                D(3, "Adding " << first << " + " << second << " = " << result);
                *(uint32_t *)(vd.data() + i) = result;
              }
            }
          }                
        } break;
        case 24: {
          //vmseq
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first == second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first == second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first == second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 25: { 
          //vmsne
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first != second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first != second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first != second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 26: {
          //vmsltu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 27: {
          //vmslt
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 28: {
          //vmsleu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 29: {
          //vmsle
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 30: {
          //vmsgtu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 31: {
          //vmsgt
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        }
        break;
      case 2: {
        switch (func6) {
        case 24: { 
          // vmandnot
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value & !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }            
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value & !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value & !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 25: {
          // vmand
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 26: {
          // vmor
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 27: { 
          //vmxor
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 28: {
          //vmornot
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value | !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value | !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value | !second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 29: {
          //vmnand
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value & second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 30: {
          //vmnor
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value | second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 31: {
          //vmxnor
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value ^ second_value);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: {
          //vmul
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 45: {
          // vmacc
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        }
      } break;
      case 6: {
        switch (func6) {
        case 0: {
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: {
          // vmul.vx
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        }
      } break;
      case 7: {
        vtype_.vill = 0;
        vtype_.vediv = instr.getVediv();
        vtype_.vsew  = instr.getVsew();
        vtype_.vlmul = instr.getVlmul();

        D(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << rsdata[0] << "VLMAX" << VLMAX);

        int s0 = rsdata[0];
        if (s0 <= VLMAX) {
          vl_ = s0;
        } else if (s0 < (2 * VLMAX)) {
          vl_ = (int)ceil((s0 * 1.0) / 2.0);
        } else if (s0 >= (2 * VLMAX)) {
          vl_ = VLMAX;
        }        
        rddata = vl_;
      } break;
      default:
        std::abort();
      }
    } break;    
    default:
      std::abort();
    }

    if (rd_write) {
      int rdt = instr.getRDType();
      switch (rdt) {
      case 1:      
        if (rdest) {
          D(2, "[" << std::dec << t << "] Dest Regs: r" << rdest << "=0x" << std::hex << std::hex << rddata);
          iregs[rdest] = rddata;
        }
        break;
      case 2:
        D(2, "[" << std::dec << t << "] Dest Regs: fr" << rdest << "=0x" << std::hex << std::hex << rddata);
        fregs[rdest] = rddata;
        break;
      default:
        break;
      }
    }
  }

  PC_ += core_->arch().wsize();
  if (PC_ != nextPC) {
    D(3, "*** Next PC: " << std::hex << nextPC << std::dec);
    PC_ = nextPC;
  }
}
