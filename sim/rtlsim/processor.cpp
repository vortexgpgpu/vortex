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

#include "processor.h"

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

#include <VX_config.h>
#include <ostream>
#include <list>
#include <queue>
#include <vector>
#include <sstream> 
#include <unordered_map>

#define RAMULATOR
#include <ramulator/src/Gem5Wrapper.h>
#include <ramulator/src/Request.h>
#include <ramulator/src/Statistics.h>

#ifndef MEMORY_BANKS
  #ifdef PLATFORM_PARAM_LOCAL_MEMORY_BANKS
    #define MEMORY_BANKS PLATFORM_PARAM_LOCAL_MEMORY_BANKS
  #else
    #define MEMORY_BANKS 2
  #endif
#endif

#ifndef MEM_CYCLE_RATIO
#define MEM_CYCLE_RATIO -1
#endif

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#ifndef VERILATOR_RESET_VALUE
#define VERILATOR_RESET_VALUE 2
#endif

#if (XLEN == 32)
typedef uint32_t Word;
#elif (XLEN == 64)
typedef uint64_t Word;
#else
#error unsupported XLEN
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
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

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

class Processor::Impl {
public:
  Impl() {
    // force random values for unitialized signals  
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // turn off assertion before reset
    Verilated::assertOn(false);

    // create RTL module instance
  #ifdef AXI_BUS
    device_ = new VVortex_axi();
  #else
    device_ = new VVortex();
  #endif

  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC();
    device_->trace(trace_, 99);
    trace_->open("trace.vcd");
  #endif

    ram_ = nullptr;
    
    // initialize dram simulator
    ramulator::Config ram_config;
    ram_config.add("standard", "DDR4");
    ram_config.add("channels", std::to_string(MEMORY_BANKS));
    ram_config.add("ranks", "1");
    ram_config.add("speed", "DDR4_2400R");
    ram_config.add("org", "DDR4_4Gb_x8");
    ram_config.add("mapping", "defaultmapping");
    ram_config.set_core_num(1);
    dram_ = new ramulator::Gem5Wrapper(ram_config, MEM_BLOCK_SIZE);
    Stats::statlist.output("ramulator.ddr4.log");

    // reset the device
    this->reset();
    
    // Turn on assertion after reset
    Verilated::assertOn(true);
  }

  ~Impl() {
    this->cout_flush();

  #ifdef VCD_OUTPUT
    trace_->close();
    delete trace_;
  #endif
    
    delete device_;
    
    if (dram_) {
      dram_->finish();
      Stats::statlist.printall();
      delete dram_;
    }
  }

  void cout_flush() {
    for (auto& buf : print_bufs_) {
      auto str = buf.second.str();
      if (!str.empty()) {
        std::cout << "#" << buf.first << ": " << str << std::endl;
      }
    }
  }

  void attach_ram(RAM* ram) {
    ram_ = ram;
  }

  int run() {
    int exitcode = 0;

  #ifndef NDEBUG
    std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
  #endif

    // start execution
    running_ = true;
    device_->reset = 0;

    // wait on device to go busy
    while (!device_->busy) {
      this->tick();
    }

    // wait on device to go idle
    while (device_->busy) {
      if (get_ebreak()) {
        exitcode = (int)get_last_wb_value(3);
        break;  
      }
      this->tick();
    }
    
    // reset device
    this->reset();

    this->cout_flush();

    return exitcode;
  }

  void write_dcr(uint32_t addr, uint32_t value) {
    device_->dcr_wr_valid = 1;
    device_->dcr_wr_addr  = addr;
    device_->dcr_wr_data  = value;
    while (device_->dcr_wr_valid) {
      this->tick();
    }
  }

private:

  void reset() {
    running_ = false;

    print_bufs_.clear();

    pending_mem_reqs_.clear();
    
    mem_rd_rsp_active_ = false;
    mem_wr_rsp_active_ = false;

  #ifdef AXI_BUS
    this->reset_axi_bus();
  #else
    this->reset_avs_bus();
  #endif

    this->reset_dcr_bus();

    device_->reset = 1;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }
  }

  void tick() {

    device_->clk = 0;
    this->eval();

  #ifdef AXI_BUS
    this->eval_axi_bus(0);
  #else
    this->eval_avs_bus(0);
  #endif
    this->eval_dcr_bus(0);

    device_->clk = 1;
    this->eval();
      
  #ifdef AXI_BUS
    this->eval_axi_bus(1);
  #else
    this->eval_avs_bus(1);
  #endif
    this->eval_dcr_bus(1);

    if (MEM_CYCLE_RATIO > 0) { 
      auto cycle = timestamp / 2;
      if ((cycle % MEM_CYCLE_RATIO) == 0)
        dram_->tick();
    } else {
      for (int i = MEM_CYCLE_RATIO; i <= 0; ++i)
        dram_->tick();            
    }

    if (!dram_queue_.empty()) {
      if (dram_->send(dram_queue_.front()))
        dram_queue_.pop();
    }

  #ifndef NDEBUG
    fflush(stdout);
  #endif
  }

  void eval() {
    device_->eval();
  #ifdef VCD_OUTPUT
    if (sim_trace_enabled()) {
      trace_->dump(timestamp);
    } else {
      exit(-1);
    }
  #endif
    ++timestamp;
  }

#ifdef AXI_BUS

  void reset_axi_bus() {    
    device_->m_axi_wready[0]  = 0;
    device_->m_axi_awready[0] = 0;
    device_->m_axi_arready[0] = 0;  
    device_->m_axi_rvalid[0]  = 0;
    device_->m_axi_bvalid[0]  = 0;
  }
    
  void eval_axi_bus(bool clk) {
    if (!clk) {
      mem_rd_rsp_ready_ = device_->m_axi_rready[0];
      mem_wr_rsp_ready_ = device_->m_axi_bready[0];
      return;
    }

    if (ram_ == nullptr) {
      device_->m_axi_wready[0]  = 0;
      device_->m_axi_awready[0] = 0;
      device_->m_axi_arready[0] = 0;  
      return;
    }

    // process memory responses
    if (mem_rd_rsp_active_
    && device_->m_axi_rvalid[0] && mem_rd_rsp_ready_) {
      mem_rd_rsp_active_ = false;
    }    
    if (!mem_rd_rsp_active_) {      
      if (!pending_mem_reqs_.empty()
       && (*pending_mem_reqs_.begin())->ready 
       && !(*pending_mem_reqs_.begin())->write) {      
        auto mem_rsp_it = pending_mem_reqs_.begin();
        auto mem_rsp = *mem_rsp_it;
        /*
          printf("%0ld: [sim] MEM Rd Rsp: bank=%d, addr=%0lx, data=", timestamp, last_mem_rsp_bank_, mem_rsp->addr);
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            printf("%02x", mem_rsp->block[(MEM_BLOCK_SIZE-1)-i]);
          }
          printf("\n");
        */      
        device_->m_axi_rvalid[0] = 1;
        device_->m_axi_rid[0]    = mem_rsp->tag;   
        device_->m_axi_rresp[0]  = 0;
        device_->m_axi_rlast[0]  = 1;
        memcpy(device_->m_axi_rdata[0].data(), mem_rsp->block.data(), MEM_BLOCK_SIZE);
        pending_mem_reqs_.erase(mem_rsp_it);
        mem_rd_rsp_active_ = true;
        delete mem_rsp;
      } else {
        device_->m_axi_rvalid[0] = 0;
      }
    }

    // send memory write response  
    if (mem_wr_rsp_active_
    && device_->m_axi_bvalid[0] && mem_wr_rsp_ready_) {
      mem_wr_rsp_active_ = false;
    }
    if (!mem_wr_rsp_active_) {
      if (!pending_mem_reqs_.empty()
       && (*pending_mem_reqs_.begin())->ready 
       && (*pending_mem_reqs_.begin())->write) {
        auto mem_rsp_it = pending_mem_reqs_.begin();
        auto mem_rsp = *mem_rsp_it;
        /*
          printf("%0ld: [sim] MEM Wr Rsp: bank=%d, addr=%0lx\n", timestamp, last_mem_rsp_bank_, mem_rsp->addr);        
        */
        device_->m_axi_bvalid[0] = 1;      
        device_->m_axi_bid[0]    = mem_rsp->tag;
        device_->m_axi_bresp[0]  = 0;
        pending_mem_reqs_.erase(mem_rsp_it);        
        mem_wr_rsp_active_ = true;
        delete mem_rsp;
      } else {
        device_->m_axi_bvalid[0] = 0;
      }      
    }

    // select the memory bank
    uint32_t req_addr = device_->m_axi_wvalid[0] ? device_->m_axi_awaddr[0] : device_->m_axi_araddr[0];
    
    // process memory requests
    if ((device_->m_axi_wvalid[0] || device_->m_axi_arvalid[0]) && running_) {
      if (device_->m_axi_wvalid[0]) {        
        uint64_t byteen = device_->m_axi_wstrb[0];
        uint64_t base_addr = device_->m_axi_awaddr[0];
        uint8_t* data = (uint8_t*)device_->m_axi_wdata[0].data();

        // check console output
        if (base_addr >= uint64_t(IO_COUT_ADDR)
         && base_addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {          
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

          auto mem_req = new mem_req_t();
          mem_req->tag   = device_->m_axi_awid[0];
          mem_req->addr  = device_->m_axi_awaddr[0];        
          mem_req->write = true;
          mem_req->ready = true;
          pending_mem_reqs_.emplace_back(mem_req);

          // send dram request
          ramulator::Request dram_req( 
            device_->m_axi_awaddr[0],
            ramulator::Request::Type::WRITE,
            0
          );
          dram_queue_.push(dram_req);
        }        
      } else {
        // process reads
        auto mem_req = new mem_req_t();
        mem_req->tag  = device_->m_axi_arid[0];
        mem_req->addr = device_->m_axi_araddr[0];
        ram_->read(mem_req->block.data(), device_->m_axi_araddr[0], MEM_BLOCK_SIZE);
        mem_req->write = false;
        mem_req->ready = false;
        pending_mem_reqs_.emplace_back(mem_req);

        // send dram request
        ramulator::Request dram_req( 
          device_->m_axi_araddr[0],
          ramulator::Request::Type::READ,
          std::bind([&](ramulator::Request& dram_req, mem_req_t* mem_req) {
              mem_req->ready = true;
            }, placeholders::_1, mem_req),
          0
        );
        dram_queue_.push(dram_req);
      } 
    } 

    device_->m_axi_wready[0]  = running_;
    device_->m_axi_awready[0] = running_;
    device_->m_axi_arready[0] = running_;     
  }

#else

  void reset_avs_bus() {
    device_->mem_req_ready = 0;
    device_->mem_rsp_valid = 0;
  }

  void eval_avs_bus(bool clk) {
    if (!clk) {
      mem_rd_rsp_ready_ = device_->mem_rsp_ready;
      return;
    }

    if (ram_ == nullptr) {
      device_->mem_req_ready = 0;
      return;
    }

    // process memory responses    
    if (mem_rd_rsp_active_
    && device_->mem_rsp_valid && mem_rd_rsp_ready_) {
      mem_rd_rsp_active_ = false;
    }
    if (!mem_rd_rsp_active_) {
      if (!pending_mem_reqs_.empty()
       && (*pending_mem_reqs_.begin())->ready) {
        device_->mem_rsp_valid = 1;      
        auto mem_rsp_it = pending_mem_reqs_.begin();
        auto mem_rsp = *mem_rsp_it;
        /*
          printf("%0ld: [sim] MEM Rd: bank=%d, tag=%0lx, addr=%0lx, data=", timestamp, last_mem_rsp_bank_, mem_rsp->tag, mem_rsp->addr);
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            printf("%02x", mem_rsp->block[(MEM_BLOCK_SIZE-1)-i]);
          }
          printf("\n");
        */
        memcpy(device_->mem_rsp_data.data(), mem_rsp->block.data(), MEM_BLOCK_SIZE);
        device_->mem_rsp_tag = mem_rsp->tag;   
        pending_mem_reqs_.erase(mem_rsp_it);
        mem_rd_rsp_active_ = true;
        delete mem_rsp;
      } else {
        device_->mem_rsp_valid = 0;
      }
    }

    // process memory requests    
    if (device_->mem_req_valid && running_) {
      uint64_t byte_addr = (device_->mem_req_addr * MEM_BLOCK_SIZE);
      if (device_->mem_req_rw) {        
        // process writes
        uint64_t byteen = device_->mem_req_byteen;        
        uint8_t* data = (uint8_t*)(device_->mem_req_data.data());

        // check console output
        if (byte_addr >= uint64_t(IO_COUT_ADDR)
         && byte_addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {    
          for (int i = 0; i < IO_COUT_SIZE; i++) {
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
            printf("%0ld: [sim] MEM Wr: tag=%0lx, addr=%0x, byteen=%0lx, data=", timestamp, device_->mem_req_tag, byte_addr, byteen);
            for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
              printf("%02x", data[(MEM_BLOCK_SIZE-1)-i]);
            }
            printf("\n");
          */
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              (*ram_)[byte_addr + i] = data[i];
            }
          }

          // send dram request
          ramulator::Request dram_req( 
            byte_addr,
            ramulator::Request::Type::WRITE,
            0
          );
          dram_queue_.push(dram_req);
        }         
      } else {
        // process reads
        auto mem_req = new mem_req_t();
        mem_req->tag   = device_->mem_req_tag;   
        mem_req->addr  = byte_addr;
        mem_req->write = false;
        mem_req->ready = false;
        ram_->read(mem_req->block.data(), byte_addr, MEM_BLOCK_SIZE);
        pending_mem_reqs_.emplace_back(mem_req);

        //printf("%0ld: [sim] MEM Rd Req: addr=%0x, tag=%0lx\n", timestamp, byte_addr, device_->mem_req_tag);

        // send dram request
        ramulator::Request dram_req( 
          byte_addr,
          ramulator::Request::Type::READ,
          std::bind([&](ramulator::Request& dram_req, mem_req_t* mem_req) {
              mem_req->ready = true;
            }, placeholders::_1, mem_req),
          0
        );
        dram_queue_.push(dram_req);
      }
    }   

    device_->mem_req_ready = running_;
  }

#endif

  void  reset_dcr_bus() {
    device_->dcr_wr_valid = 0;
  }

  void  eval_dcr_bus(bool clk) {
    if (!clk) {
      return;
    }
    if (device_->dcr_wr_valid) {
      device_->dcr_wr_valid = 0;
    }
  }

  void wait(uint32_t cycles) {
    for (int i = 0; i < cycles; ++i) {
      this->tick();
    }
  }

  bool get_ebreak() const {
  #ifdef AXI_BUS
    return (bool)device_->Vortex_axi->vortex->sim_ebreak;
  #else
    return (bool)device_->Vortex->sim_ebreak;
  #endif
  }

  uint64_t get_last_wb_value(int reg) const {
  #ifdef AXI_BUS
    return ((Word*)device_->Vortex_axi->vortex->sim_wb_value.data())[reg];
  #else
    return ((Word*)device_->Vortex->sim_wb_value.data())[reg];
  #endif
  }

private:

  typedef struct {    
    bool ready;  
    std::array<uint8_t, MEM_BLOCK_SIZE> block;
    uint64_t addr;
    uint64_t tag;
    bool write;
  } mem_req_t;

#ifdef AXI_BUS
  VVortex_axi *device_;
#else
  VVortex *device_;
#endif
#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif

  std::unordered_map<int, std::stringstream> print_bufs_;

  std::list<mem_req_t*> pending_mem_reqs_;

  bool mem_rd_rsp_active_;
  bool mem_rd_rsp_ready_;

  bool mem_wr_rsp_active_;
  bool mem_wr_rsp_ready_;

  RAM *ram_;

  ramulator::Gem5Wrapper* dram_;

  std::queue<ramulator::Request> dram_queue_;

  bool running_;
};

///////////////////////////////////////////////////////////////////////////////

Processor::Processor() 
  : impl_(new Impl())
{}

Processor::~Processor() {
  delete impl_;
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

int Processor::run() {
  return impl_->run();
}

void Processor::write_dcr(uint32_t addr, uint32_t value) {
  return impl_->write_dcr(addr, value);
}