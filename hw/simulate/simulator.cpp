#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

Simulator::Simulator() {  
  // force random values for unitialized signals  
  Verilated::randReset(2);

  // Turn off assertion before reset
  Verilated::assertOn(false);

  ram_ = nullptr;
  vortex_ = new VVortex();

  dram_rsp_active_ = false;
  snp_req_active_ = false;

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC();
  trace_->set_time_unit("1ns");
  vortex_->trace(trace_, 99);
  trace_->open("trace.vcd");
#endif  
}

Simulator::~Simulator() {
#ifdef VCD_OUTPUT
  trace_->close();
#endif
  delete vortex_;
}

void Simulator::attach_ram(RAM* ram) {
  ram_ = ram;
  dram_rsp_vec_.clear();
}

void Simulator::reset() {     
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] reset()" << std::endl;
#endif

  vortex_->reset = 1;
  this->step();  
  vortex_->reset = 0;

  dram_rsp_vec_.clear();

  // Turn on assertion after reset
  Verilated::assertOn(true);
}

void Simulator::step() {
  vortex_->clk = 0;
  this->eval();

  vortex_->clk = 1;
  this->eval();

  this->eval_dram_bus();
  this->eval_io_bus();
  this->eval_csr_bus();
  this->eval_snp_bus();
}

void Simulator::eval() {
  vortex_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void Simulator::eval_dram_bus() {
  if (ram_ == nullptr) {
    vortex_->dram_req_ready = 0;
    return;
  }

  // schedule DRAM responses
  int dequeue_index = -1;
  for (int i = 0; i < dram_rsp_vec_.size(); i++) {
    if (dram_rsp_vec_[i].cycles_left > 0) {
      dram_rsp_vec_[i].cycles_left -= 1;
    }
    if ((dequeue_index == -1) 
     && (dram_rsp_vec_[i].cycles_left == 0)) {
      dequeue_index = i;
    }
  }

  // send DRAM response  
  if (dram_rsp_active_
   && vortex_->dram_rsp_valid 
   && vortex_->dram_rsp_ready) {
    dram_rsp_active_ = false;
  }
  if (!dram_rsp_active_) {
    if (dequeue_index != -1) {
      vortex_->dram_rsp_valid = 1;
      memcpy((uint8_t*)vortex_->dram_rsp_data, dram_rsp_vec_[dequeue_index].block.data(), GLOBAL_BLOCK_SIZE);
      vortex_->dram_rsp_tag = dram_rsp_vec_[dequeue_index].tag;   
      dram_rsp_vec_.erase(dram_rsp_vec_.begin() + dequeue_index);
      dram_rsp_active_ = true;
    } else {
      vortex_->dram_rsp_valid = 0;
    }
  }

  // handle DRAM stalls
  bool dram_stalled = false;
#ifdef ENABLE_DRAM_STALLS
  if (0 == ((timestamp/2) % DRAM_STALLS_MODULO)) { 
    dram_stalled = true;
  } else
  if (dram_rsp_vec_.size() >= DRAM_RQ_SIZE) {
    dram_stalled = true;
  }
#endif

  // process DRAM requests
  if (!dram_stalled) {
    if (vortex_->dram_req_valid) {
      if (vortex_->dram_req_rw) {
        uint64_t byteen = vortex_->dram_req_byteen;
        unsigned base_addr = (vortex_->dram_req_addr * GLOBAL_BLOCK_SIZE);
        uint8_t* data = (uint8_t*)(vortex_->dram_req_data);
        for (int i = 0; i < GLOBAL_BLOCK_SIZE; i++) {
          if ((byteen >> i) & 0x1) {            
            (*ram_)[base_addr + i] = data[i];
          }
        }
      } else {
        dram_req_t dram_req;
        dram_req.cycles_left = DRAM_LATENCY;     
        dram_req.tag = vortex_->dram_req_tag;
        ram_->read(vortex_->dram_req_addr * GLOBAL_BLOCK_SIZE, GLOBAL_BLOCK_SIZE, dram_req.block.data());
        dram_rsp_vec_.push_back(dram_req);
      } 
    }    
  }

  vortex_->dram_req_ready = ~dram_stalled;
}

void Simulator::eval_io_bus() {
  if (vortex_->io_req_valid
   && vortex_->io_req_rw 
   && ((vortex_->io_req_addr << 2) == IO_BUS_ADDR_COUT)) {
    uint32_t data_write = (uint32_t)vortex_->io_req_data;
    char c = (char)data_write;
    std::cout << c;      
  }
  vortex_->io_req_ready = 1;
  vortex_->io_rsp_valid = 0;
}

void Simulator::eval_csr_bus() {
  vortex_->csr_io_req_valid = 0;
  vortex_->csr_io_req_coreid = 0;
  vortex_->csr_io_req_addr = 0;
  vortex_->csr_io_req_rw = 0;
  vortex_->csr_io_req_data = 0;  
  vortex_->csr_io_rsp_ready = 1;
}

void Simulator::eval_snp_bus() {
  if (snp_req_active_) {       
    if (vortex_->snp_rsp_valid) {      
      assert(pending_snp_reqs_ > 0);
      --pending_snp_reqs_;
    #ifdef DBG_PRINT_CACHE_SNP
      std::cout << timestamp << ": [sim] snp rsp: tag=" << vortex_->snp_rsp_tag << " pending=" << pending_snp_reqs_ << std::endl;
    #endif
    }
    if (vortex_->snp_req_valid && vortex_->snp_req_ready) {            
      if (snp_req_size_) {        
        vortex_->snp_req_addr += 1;
        vortex_->snp_req_tag  += 1;
        --snp_req_size_;
        ++pending_snp_reqs_;
      #ifdef DBG_PRINT_CACHE_SNP
        std::cout << timestamp << ": [sim] snp req: addr=" << std::hex << vortex_->snp_req_addr << std::dec << " tag=" << vortex_->snp_req_tag << " remain=" << snp_req_size_ << std::endl;
      #endif
      } else {
        vortex_->snp_req_valid = 0;        
      }      
    }
    if (!vortex_->snp_req_valid 
     && 0 == pending_snp_reqs_) {
      snp_req_active_ = false;
    }  
  } else {
    vortex_->snp_req_valid = 0;
    vortex_->snp_rsp_ready = 0;
  }
}

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() const {
  return vortex_->busy || snp_req_active_;
}

void Simulator::flush_caches(uint32_t mem_addr, uint32_t size) {  
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] flush_caches()" << std::endl;
#endif
  if (0 == size)
    return;

  snp_req_active_ = true;
  snp_req_size_   = (size + GLOBAL_BLOCK_SIZE - 1) / GLOBAL_BLOCK_SIZE; 
  
  vortex_->snp_req_addr  = mem_addr / GLOBAL_BLOCK_SIZE;
  vortex_->snp_req_tag   = 0;
  vortex_->snp_req_valid = 1;
  vortex_->snp_rsp_ready = 1;  

  --snp_req_size_;
  pending_snp_reqs_ = 1;
    
  #ifdef DBG_PRINT_CACHE_SNP
    std::cout << timestamp << ": [sim] snp req: addr=" << std::hex << vortex_->snp_req_addr << std::dec << " tag=" << vortex_->snp_req_tag << " remain=" << snp_req_size_ << std::endl;
  #endif  
}

void Simulator::run() {
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] run()" << std::endl;
#endif 

  // reset the device
  this->reset();

  // execute program
  while (vortex_->busy 
      && !vortex_->ebreak) {
    this->step();
  }

  // wait 5 cycles to flush the pipeline
  this->wait(5);  
}

int Simulator::get_last_wb_value(int reg) const {
  #if (NUM_CLUSTERS != 1)
    return (int)vortex_->Vortex->genblk2__DOT__genblk1__BRA__0__KET____DOT__cluster->genblk1__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];    
  #else
    return (int)vortex_->Vortex->genblk1__DOT__cluster->genblk1__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];
  #endif
}

void Simulator::load_bin(const char* program_file) {
  if (ram_ == nullptr)
    return;

  std::ifstream ifs(program_file);
  if (!ifs) {
    std::cout << "error: " << program_file << " not found" << std::endl;
  }

  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  std::vector<uint8_t> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read((char*)content.data(), size);

  ram_->write(STARTUP_ADDR, size, content.data());
}

void Simulator::load_ihex(const char* program_file) {
  if (ram_ == nullptr)
    return;

  auto hti = [&](char c)->uint32_t {
    if (c >= 'A' && c <= 'F')
      return c - 'A' + 10;
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    return c - '0';
  };

  auto hToI = [&](const char *c, uint32_t size)->uint32_t {
    uint32_t value = 0;
    for (uint32_t i = 0; i < size; i++) {
      value += hti(c[i]) << ((size - i - 1) * 4);
    }
    return value;
  };

  std::ifstream ifs(program_file);
  if (!ifs) {
    std::cout << "error: " << program_file << " not found" << std::endl;
  }

  ifs.seekg(0, ifs.end);
  uint32_t size = ifs.tellg();
  std::vector<char> content(size);
  ifs.seekg(0, ifs.beg);
  ifs.read(content.data(), size);

  int offset = 0;
  char *line = content.data();

  while (true) {
    if (line[0] == ':') {
      uint32_t byteCount = hToI(line + 1, 2);
      uint32_t nextAddr = hToI(line + 3, 4) + offset;
      uint32_t key = hToI(line + 7, 2);
      switch (key) {
      case 0:
        for (uint32_t i = 0; i < byteCount; i++) {
          (*ram_)[nextAddr + i] = hToI(line + 9 + i * 2, 2);
        }
        break;
      case 2:
        offset = hToI(line + 9, 4) << 4;
        break;
      case 4:
        offset = hToI(line + 9, 4) << 16;
        break;
      default:
        break;
      }
    }
    while (*line != '\n' && size != 0) {
      ++line;
      --size;
    }
    if (size <= 1)
      break;
    ++line;
    --size;
  }
}

void Simulator::print_stats(std::ostream& out) {
  out << std::left;
  out << std::setw(24) << "# of total cycles:" << std::dec << timestamp/2 << std::endl;
}