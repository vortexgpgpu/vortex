#include "simulator.h"

#include <verilated.h>

#ifdef AXI_BUS
#include "VVortex_axi.h"
#include "VVortex_axi__Syms.h"
#else
#include "VVortex.h"
#include "VVortex__Syms.h"
#endif

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <mem.h>

#define ENABLE_MEM_STALLS

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#ifndef MEM_LATENCY
#define MEM_LATENCY 24
#endif

#ifndef MEM_RQ_SIZE
#define MEM_RQ_SIZE 16
#endif

#ifndef MEM_STALLS_MODULO
#define MEM_STALLS_MODULO 16
#endif

#ifndef VERILATOR_RESET_VALUE
#define VERILATOR_RESET_VALUE 2
#endif

#define VL_WDATA_GETW(lwp, i, n, w) \
  VL_SEL_IWII(0, n * w, 0, 0, lwp, i * w, w)

using namespace vortex;

static uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

///////////////////////////////////////////////////////////////////////////////

static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time = TRACE_STOP_TIME;

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time 
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

///////////////////////////////////////////////////////////////////////////////

namespace vortex {
class VL_OBJ {
public:
#ifdef AXI_BUS
  VVortex_axi *device;
#else
  VVortex *device;
#endif
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace;
#endif

  VL_OBJ() {
    // force random values for unitialized signals  
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // Turn off assertion before reset
    Verilated::assertOn(false);

  #ifdef AXI_BUS
    this->device = new VVortex_axi();
  #else
    this->device = new VVortex();
  #endif

  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    this->trace = new VerilatedVcdC();
    this->device->trace(this->trace, 99);
    this->trace->open("trace.vcd");
  #endif
  }

  ~VL_OBJ() {
  #ifdef VCD_OUTPUT
    this->trace->close();
    delete this->trace;
  #endif
    delete this->device;
  }
};
}

///////////////////////////////////////////////////////////////////////////////

Simulator::Simulator() {
  vl_obj_ = new VL_OBJ();
  ram_ = nullptr;
  // reset the device
  this->reset();
}

Simulator::~Simulator() {
  for (auto& buf : print_bufs_) {
    auto str = buf.second.str();
    if (!str.empty()) {
      std::cout << "#" << buf.first << ": " << str << std::endl;
    }
  }
  delete vl_obj_;
}

void Simulator::attach_ram(RAM* ram) {
  ram_ = ram;
  for (int b = 0; b < MEMORY_BANKS; ++b) {
    mem_rsp_vec_[b].clear();
  }
  last_mem_rsp_bank_ = 0;
}

void Simulator::reset() { 
  print_bufs_.clear();

  for (int b = 0; b < MEMORY_BANKS; ++b) {
    mem_rsp_vec_[b].clear();
  }
  last_mem_rsp_bank_ = 0;
  mem_rd_rsp_active_ = false;
  mem_wr_rsp_active_ = false;

#ifdef AXI_BUS
  this->reset_axi_bus();
#else
  this->reset_mem_bus();
#endif

  vl_obj_->device->reset = 1;

  for (int i = 0; i < RESET_DELAY; ++i) {
    vl_obj_->device->clk = 0;
    this->eval();
    vl_obj_->device->clk = 1;
    this->eval();
  }  

  vl_obj_->device->reset = 0;
  
  // Turn on assertion after reset
  Verilated::assertOn(true);
}

void Simulator::step() {

  vl_obj_->device->clk = 0;
  this->eval();

#ifdef AXI_BUS
  this->eval_axi_bus(0);
#else
  this->eval_mem_bus(0);
#endif

  vl_obj_->device->clk = 1;
  this->eval();
    
#ifdef AXI_BUS
  this->eval_axi_bus(1);
#else
  this->eval_mem_bus(1);
#endif

#ifndef NDEBUG
  fflush(stdout);
#endif
}

void Simulator::eval() {
  vl_obj_->device->eval();
#ifdef VCD_OUTPUT
  if (sim_trace_enabled()) {
    vl_obj_->trace->dump(timestamp);
  }
#endif
  ++timestamp;
}

#ifdef AXI_BUS

void Simulator::reset_axi_bus() {
  vl_obj_->device->m_axi_wready  = 0;
  vl_obj_->device->m_axi_awready = 0;
  vl_obj_->device->m_axi_arready = 0;  
  vl_obj_->device->m_axi_rvalid  = 0;
  vl_obj_->device->m_axi_bvalid  = 0;
}
  
void Simulator::eval_axi_bus(bool clk) {
  if (!clk) {
    mem_rd_rsp_ready_ = vl_obj_->device->m_axi_rready;
    mem_wr_rsp_ready_ = vl_obj_->device->m_axi_bready;
    return;
  }

  if (ram_ == nullptr) {
    vl_obj_->device->m_axi_wready  = 0;
    vl_obj_->device->m_axi_awready = 0;
    vl_obj_->device->m_axi_arready = 0;  
    return;
  }

  // update memory responses schedule
  for (int b = 0; b < MEMORY_BANKS; ++b) {    
    for (auto& rsp : mem_rsp_vec_[b]) {
      if (rsp.cycles_left > 0)
        rsp.cycles_left -= 1;
    }
  }

  bool has_rd_response = false;
  bool has_wr_response = false;

  // schedule memory responses that are ready
  for (int i = 0; i < MEMORY_BANKS; ++i) {
    uint32_t b = (i + last_mem_rsp_bank_ + 1) % MEMORY_BANKS;
    if (!mem_rsp_vec_[b].empty()) {
      auto mem_rsp_it = mem_rsp_vec_[b].begin();
      if (mem_rsp_it->cycles_left <= 0) {
          has_rd_response = !mem_rsp_it->write;
          has_wr_response = mem_rsp_it->write;
          last_mem_rsp_bank_ = b;
          break;
      }
    }
  }

  // send memory read response  
  if (mem_rd_rsp_active_
  && vl_obj_->device->m_axi_rvalid && mem_rd_rsp_ready_) {
    mem_rd_rsp_active_ = false;
  }
  if (!mem_rd_rsp_active_) {
    if (has_rd_response) {      
      auto mem_rsp_it = mem_rsp_vec_[last_mem_rsp_bank_].begin();
      /*
        printf("%0ld: [sim] MEM Rd Rsp: bank=%d, addr=%0lx, data=", timestamp, last_mem_rsp_bank_, mem_rsp_it->addr);
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          printf("%02x", mem_rsp_it->block[(MEM_BLOCK_SIZE-1)-i]);
        }
        printf("\n");
      */      
      vl_obj_->device->m_axi_rvalid = 1;
      vl_obj_->device->m_axi_rid    = mem_rsp_it->tag;   
      vl_obj_->device->m_axi_rresp  = 0;
      vl_obj_->device->m_axi_rlast  = 1;
      memcpy((uint8_t*)vl_obj_->device->m_axi_rdata, mem_rsp_it->block.data(), MEM_BLOCK_SIZE);
      mem_rsp_vec_[last_mem_rsp_bank_].erase(mem_rsp_it);
      mem_rd_rsp_active_ = true;
    } else {
      vl_obj_->device->m_axi_rvalid = 0;
    }
  }

  // send memory write response  
  if (mem_wr_rsp_active_
  && vl_obj_->device->m_axi_bvalid && mem_wr_rsp_ready_) {
    mem_wr_rsp_active_ = false;
  }
  if (!mem_wr_rsp_active_) {
    if (has_wr_response) {
      auto mem_rsp_it = mem_rsp_vec_[last_mem_rsp_bank_].begin();
      /*
        printf("%0ld: [sim] MEM Wr Rsp: bank=%d, addr=%0lx\n", timestamp, last_mem_rsp_bank_, mem_rsp_it->addr);        
      */
      vl_obj_->device->m_axi_bvalid = 1;      
      vl_obj_->device->m_axi_bid    = mem_rsp_it->tag;
      vl_obj_->device->m_axi_bresp  = 0;
      mem_rsp_vec_[last_mem_rsp_bank_].erase(mem_rsp_it);
      mem_wr_rsp_active_ = true;
    } else {
      vl_obj_->device->m_axi_bvalid = 0;
    }
  }

  // select the memory bank
  uint32_t req_addr = vl_obj_->device->m_axi_wvalid ? vl_obj_->device->m_axi_awaddr : vl_obj_->device->m_axi_araddr;
  uint32_t req_bank = (MEMORY_BANKS >= 2) ? ((req_addr / MEM_BLOCK_SIZE) % MEMORY_BANKS) : 0;

  // handle memory stalls
  bool mem_stalled = false;
#ifdef ENABLE_MEM_STALLS
  if (0 == ((timestamp/2) % MEM_STALLS_MODULO)) { 
    mem_stalled = true;
  } else
  if (mem_rsp_vec_[req_bank].size() >= MEM_RQ_SIZE) {
    mem_stalled = true;
  }
#endif

  // process memory requests
  if (!mem_stalled) {
    if (vl_obj_->device->m_axi_wvalid || vl_obj_->device->m_axi_arvalid) {
      if (vl_obj_->device->m_axi_wvalid) {        
        uint64_t byteen = vl_obj_->device->m_axi_wstrb;
        unsigned base_addr = vl_obj_->device->m_axi_awaddr;
        uint8_t* data = (uint8_t*)(vl_obj_->device->m_axi_wdata);

        // detect stdout write
        if (base_addr >= IO_COUT_ADDR 
         && base_addr <= (IO_COUT_ADDR + IO_COUT_SIZE - 1)) {          
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              auto& ss_buf = print_bufs_[i];
              char c = data[i];
              ss_buf << c;
              if (c == '\n') {
                std::cout << std::dec << "#" << i << ": " << ss_buf.str() << std::flush;
                ss_buf.str("");
              }
            }
          }   
        } else {
          /*
            printf("%0ld: [sim] MEM Wr: addr=%0x, byteen=%0lx, data=", timestamp, base_addr, byteen);
            for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
              printf("%02x", data[(MEM_BLOCK_SIZE-1)-i]);
            }
            printf("\n");
          */
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              (*ram_)[base_addr + i] = data[i];
            }
          }
          mem_req_t mem_req;
          mem_req.tag  = vl_obj_->device->m_axi_arid;
          mem_req.addr = vl_obj_->device->m_axi_araddr;        
          mem_req.cycles_left = 0;
          mem_req.write = 1;
          mem_rsp_vec_[req_bank].emplace_back(mem_req);
        }        
      } else {
        mem_req_t mem_req;        
        mem_req.tag  = vl_obj_->device->m_axi_arid;   
        mem_req.addr = vl_obj_->device->m_axi_araddr;
        ram_->read(mem_req.block.data(), vl_obj_->device->m_axi_araddr, MEM_BLOCK_SIZE);
        mem_req.cycles_left = MEM_LATENCY;
        mem_req.write = 0;
        for (auto& rsp : mem_rsp_vec_[req_bank]) {
          if (mem_req.addr == rsp.addr) {
            // duplicate requests receive the same cycle delay
            mem_req.cycles_left = rsp.cycles_left;
            break;
          }
        }     
        mem_rsp_vec_[req_bank].emplace_back(mem_req);
      } 
    }    
  }

  vl_obj_->device->m_axi_wready  = !mem_stalled;
  vl_obj_->device->m_axi_awready = !mem_stalled;
  vl_obj_->device->m_axi_arready = !mem_stalled;
}

#else

void Simulator::reset_mem_bus() {
  vl_obj_->device->mem_req_ready = 0;
  vl_obj_->device->mem_rsp_valid = 0;
}

void Simulator::eval_mem_bus(bool clk) {
  if (!clk) {
    mem_rd_rsp_ready_ = vl_obj_->device->mem_rsp_ready;
    return;
  }

  if (ram_ == nullptr) {
    vl_obj_->device->mem_req_ready = 0;
    return;
  }

  // update memory responses schedule
  for (int b = 0; b < MEMORY_BANKS; ++b) {    
    for (auto& rsp : mem_rsp_vec_[b]) {
      if (rsp.cycles_left > 0)
        rsp.cycles_left -= 1;
    }
  }

  bool has_response = false;

  // schedule memory responses that are ready
  for (int i = 0; i < MEMORY_BANKS; ++i) {
    uint32_t b = (i + last_mem_rsp_bank_ + 1) % MEMORY_BANKS;
    if (!mem_rsp_vec_[b].empty()
    && (mem_rsp_vec_[b].begin()->cycles_left) <= 0) {
        has_response = true;
        last_mem_rsp_bank_ = b;
        break;
    }
  }

  // send memory response  
  if (mem_rd_rsp_active_
  && vl_obj_->device->mem_rsp_valid && mem_rd_rsp_ready_) {
    mem_rd_rsp_active_ = false;
  }
  if (!mem_rd_rsp_active_) {
    if (has_response) {
      vl_obj_->device->mem_rsp_valid = 1;      
      auto mem_rsp_it = mem_rsp_vec_[last_mem_rsp_bank_].begin();      
      /*
        printf("%0ld: [sim] MEM Rd: bank=%d, addr=%0lx, data=", timestamp, last_mem_rsp_bank_, mem_rsp_it->addr);
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          printf("%02x", mem_rsp_it->block[(MEM_BLOCK_SIZE-1)-i]);
        }
        printf("\n");
      */
      memcpy((uint8_t*)vl_obj_->device->mem_rsp_data, mem_rsp_it->block.data(), MEM_BLOCK_SIZE);
      vl_obj_->device->mem_rsp_tag = mem_rsp_it->tag;   
      mem_rsp_vec_[last_mem_rsp_bank_].erase(mem_rsp_it);
      mem_rd_rsp_active_ = true;
    } else {
      vl_obj_->device->mem_rsp_valid = 0;
    }
  }

  // select the memory bank
  uint32_t req_bank = (MEMORY_BANKS >= 2) ? (vl_obj_->device->mem_req_addr % MEMORY_BANKS) : 0;

  // handle memory stalls
  bool mem_stalled = false;
#ifdef ENABLE_MEM_STALLS
  if (0 == ((timestamp/2) % MEM_STALLS_MODULO)) { 
    mem_stalled = true;
  } else
  if (mem_rsp_vec_[req_bank].size() >= MEM_RQ_SIZE) {
    mem_stalled = true;
  }
#endif

  // process memory requests
  if (!mem_stalled) {
    if (vl_obj_->device->mem_req_valid) {
      if (vl_obj_->device->mem_req_rw) {        
        uint64_t byteen = vl_obj_->device->mem_req_byteen;
        unsigned base_addr = (vl_obj_->device->mem_req_addr * MEM_BLOCK_SIZE);
        uint8_t* data = (uint8_t*)(vl_obj_->device->mem_req_data);
        if (base_addr >= IO_COUT_ADDR 
         && base_addr <= (IO_COUT_ADDR + IO_COUT_SIZE - 1)) {          
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              auto& ss_buf = print_bufs_[i];
              char c = data[i];
              ss_buf << c;
              if (c == '\n') {
                std::cout << std::dec << "#" << i << ": " << ss_buf.str() << std::flush;
                ss_buf.str("");
              }
            }
          }   
        } else {
          /*
            printf("%0ld: [sim] MEM Wr: addr=%0x, byteen=%0lx, data=", timestamp, base_addr, byteen);
            for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
              printf("%02x", data[(MEM_BLOCK_SIZE-1)-i]);
            }
            printf("\n");
          */
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              (*ram_)[base_addr + i] = data[i];
            }
          }
        }
      } else {
        mem_req_t mem_req;        
        mem_req.tag  = vl_obj_->device->mem_req_tag;   
        mem_req.addr = (vl_obj_->device->mem_req_addr * MEM_BLOCK_SIZE);
        ram_->read(mem_req.block.data(), vl_obj_->device->mem_req_addr * MEM_BLOCK_SIZE, MEM_BLOCK_SIZE);
        mem_req.cycles_left = MEM_LATENCY;
        for (auto& rsp : mem_rsp_vec_[req_bank]) {
          if (mem_req.addr == rsp.addr) {
            // duplicate requests receive the same cycle delay
            mem_req.cycles_left = rsp.cycles_left;
            break;
          }
        }     
        mem_rsp_vec_[req_bank].emplace_back(mem_req);
      } 
    }    
  }

  vl_obj_->device->mem_req_ready = !mem_stalled;
}

#endif

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() const {
  return vl_obj_->device->busy;
}

int Simulator::run() {
  int exitcode = 0;

#ifndef NDEBUG
  std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
#endif

  // execute program
  while (vl_obj_->device->busy) {
    if (get_ebreak()) {
      exitcode = get_last_wb_value(3);
      break;  
    }
    this->step();
  }

  // wait 5 cycles to flush the pipeline
  this->wait(5);  

  return exitcode;
}

bool Simulator::get_ebreak() const {
#ifdef AXI_BUS
  return (int)vl_obj_->device->Vortex_axi->vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->execute->ebreak;
#else
  return (int)vl_obj_->device->Vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->execute->ebreak;
#endif
}

int Simulator::get_last_wb_value(int reg) const {
#ifdef AXI_BUS
  return (int)vl_obj_->device->Vortex_axi->vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];
#else
  return (int)vl_obj_->device->Vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];
#endif
}

void Simulator::print_stats(std::ostream& out) {
  out << std::left;
  out << std::setw(24) << "# of total cycles:" << std::dec << timestamp/2 << std::endl;
}