#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define RESET_DELAY 4

#define ENABLE_MEM_STALLS

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

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

Simulator::Simulator() {  
  // force random values for unitialized signals  
  Verilated::randReset(VERILATOR_RESET_VALUE);
  Verilated::randSeed(50);

  // Turn off assertion before reset
  Verilated::assertOn(false);

  ram_ = nullptr;
  vortex_ = new VVortex();

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC();
  vortex_->trace(trace_, 99);
  trace_->open("trace.vcd");
#endif  

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
#ifdef VCD_OUTPUT
  trace_->close();
#endif
  delete vortex_;
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

  mem_rsp_active_ = false;

  vortex_->mem_rsp_valid = 0;
  vortex_->mem_req_ready = 0;

  vortex_->reset = 1;

  for (int i = 0; i < RESET_DELAY; ++i) {
    vortex_->clk = 0;
    this->eval();
    vortex_->clk = 1;
    this->eval();
  }  

  vortex_->reset = 0;
  
  // Turn on assertion after reset
  Verilated::assertOn(true);
}

void Simulator::step() {

  vortex_->clk = 0;
  this->eval();

  mem_rsp_ready_ = vortex_->mem_rsp_ready;
  
  vortex_->clk = 1;
  this->eval();
    
  this->eval_mem_bus();

#ifndef NDEBUG
  fflush(stdout);
#endif
}

void Simulator::eval() {
  vortex_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void Simulator::eval_mem_bus() {
  if (ram_ == nullptr) {
    vortex_->mem_req_ready = 0;
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

  // schedule memory responses in FIFO order
  for (int i = 0; i < MEMORY_BANKS; ++i) {
    uint32_t b = (i + last_mem_rsp_bank_ + 1) % MEMORY_BANKS;
    if (!mem_rsp_vec_[b].empty()
    && (0 == mem_rsp_vec_[b].begin()->cycles_left)) {
        has_response = true;
        last_mem_rsp_bank_ = b;
        break;
    }
  }

  // send memory response  
  if (mem_rsp_active_
  && vortex_->mem_rsp_valid && mem_rsp_ready_) {
    mem_rsp_active_ = false;
  }
  if (!mem_rsp_active_) {
    if (has_response) {
      vortex_->mem_rsp_valid = 1;      
      std::list<mem_req_t>::iterator mem_rsp_it = mem_rsp_vec_[last_mem_rsp_bank_].begin();
      memcpy((uint8_t*)vortex_->mem_rsp_data, mem_rsp_it->block.data(), MEM_BLOCK_SIZE);
      vortex_->mem_rsp_tag = mem_rsp_it->tag;   
      mem_rsp_vec_[last_mem_rsp_bank_].erase(mem_rsp_it);
      mem_rsp_active_ = true;
    } else {
      vortex_->mem_rsp_valid = 0;
    }
  }

  // select the memory bank
  uint32_t req_bank = (MEMORY_BANKS >= 2) ? (vortex_->mem_req_addr % MEMORY_BANKS) : 0;

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
    if (vortex_->mem_req_valid) {
      if (vortex_->mem_req_rw) {        
        uint64_t byteen = vortex_->mem_req_byteen;
        unsigned base_addr = (vortex_->mem_req_addr * MEM_BLOCK_SIZE);
        uint8_t* data = (uint8_t*)(vortex_->mem_req_data);
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
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {            
              (*ram_)[base_addr + i] = data[i];
            }
          }
        }
      } else {        
        mem_req_t mem_req;        
        mem_req.tag  = vortex_->mem_req_tag;   
        mem_req.addr = vortex_->mem_req_addr;
        ram_->read(vortex_->mem_req_addr * MEM_BLOCK_SIZE, MEM_BLOCK_SIZE, mem_req.block.data());
        mem_req.cycles_left = MEM_LATENCY;
        for (auto& rsp : mem_rsp_vec_[req_bank]) {
          if (mem_req.addr == rsp.addr) {
            mem_req.cycles_left = rsp.cycles_left;
            break;
          }
        }     
        mem_rsp_vec_[req_bank].emplace_back(mem_req);
      } 
    }    
  }

  vortex_->mem_req_ready = !mem_stalled;
}

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() const {
  return vortex_->busy;
}

int Simulator::run() {
  int exitcode = 0;

#ifndef NDEBUG
  std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
#endif

  // execute program
  while (vortex_->busy) {
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
  return (int)vortex_->Vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->execute->ebreak;
}

int Simulator::get_last_wb_value(int reg) const {
  return (int)vortex_->Vortex->genblk2__BRA__0__KET____DOT__cluster->genblk2__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];
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