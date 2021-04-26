#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define RESET_DELAY 4

#define ENABLE_MEM_STALLS
#define MEM_LATENCY 24
#define MEM_RQ_SIZE 16
#define MEM_STALLS_MODULO 16

#define VL_WDATA_GETW(lwp, i, n, w) \
  VL_SEL_IWII(0, n * w, 0, 0, lwp, i * w, w)

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

Simulator::Simulator() {  
  // force random values for unitialized signals  
  Verilated::randReset(2);
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
    if (str.size()) {
      std::cout << "#" << buf.first << ": " << buf.second.str() << std::endl;
    }
  }
#ifdef VCD_OUTPUT
  trace_->close();
#endif
  delete vortex_;
}

void Simulator::attach_ram(RAM* ram) {
  ram_ = ram;
  mem_rsp_vec_.clear();
}

void Simulator::reset() { 
  print_bufs_.clear();
  mem_rsp_vec_.clear();

  mem_rsp_active_ = false;
  csr_req_active_ = false;
  csr_rsp_value_ = nullptr;

  vortex_->mem_rsp_valid = 0;
  vortex_->mem_req_ready = 0;
  //vortex_->io_req_ready = 0;
  //vortex_->io_rsp_valid = 0;
  vortex_->csr_req_valid  = 0;
  vortex_->csr_rsp_ready  = 0;

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
  csr_req_ready_ = vortex_->csr_req_ready;
  
  vortex_->clk = 1;
  this->eval();
    
  this->eval_mem_bus();
  this->eval_io_bus();
  this->eval_csr_bus();

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
  for (auto& rsp : mem_rsp_vec_) {
    if (rsp.cycles_left > 0)
      rsp.cycles_left -= 1;
  }

  // schedule memory responses in FIFO order
  std::list<mem_req_t>::iterator mem_rsp_it(mem_rsp_vec_.end());
  if (!mem_rsp_vec_.empty() 
   && (0 == mem_rsp_vec_.begin()->cycles_left)) {
      mem_rsp_it = mem_rsp_vec_.begin();
  }

  // send memory response  
  if (mem_rsp_active_
   && vortex_->mem_rsp_valid && mem_rsp_ready_) {
    mem_rsp_active_ = false;
  }
  if (!mem_rsp_active_) {
    if (mem_rsp_it != mem_rsp_vec_.end()) {
      vortex_->mem_rsp_valid = 1;
      memcpy((uint8_t*)vortex_->mem_rsp_data, mem_rsp_it->block.data(), MEM_BLOCK_SIZE);
      vortex_->mem_rsp_tag = mem_rsp_it->tag;   
      mem_rsp_vec_.erase(mem_rsp_it);
      mem_rsp_active_ = true;
    } else {
      vortex_->mem_rsp_valid = 0;
    }
  }

  // handle memory stalls
  bool mem_stalled = false;
#ifdef ENABLE_MEM_STALLS
  if (0 == ((timestamp/2) % MEM_STALLS_MODULO)) { 
    mem_stalled = true;
  } else
  if (mem_rsp_vec_.size() >= MEM_RQ_SIZE) {
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
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          if ((byteen >> i) & 0x1) {            
            (*ram_)[base_addr + i] = data[i];
          }
        }
      } else {
        mem_req_t mem_req;        
        mem_req.tag  = vortex_->mem_req_tag;   
        mem_req.addr = vortex_->mem_req_addr;
        ram_->read(vortex_->mem_req_addr * MEM_BLOCK_SIZE, MEM_BLOCK_SIZE, mem_req.block.data());
        mem_req.cycles_left = MEM_LATENCY;
        for (auto& rsp : mem_rsp_vec_) {
          if (mem_req.addr == rsp.addr) {
            mem_req.cycles_left = rsp.cycles_left;
            break;
          }
        }     
        mem_rsp_vec_.emplace_back(mem_req);
      } 
    }    
  }

  vortex_->mem_req_ready = !mem_stalled;
}

void Simulator::eval_io_bus() {
  /*for (int i = 0; i < NUM_THREADS; ++i) {
    if (((vortex_->io_req_valid >> i) & 0x1) 
     && ((VL_WDATA_GETW(vortex_->io_req_addr, i, NUM_THREADS, 30) << 2) == IO_BUS_ADDR_COUT)) {
      assert(vortex_->io_req_rw);
      int data = vortex_->io_req_data[i];
      int tid = data >> 16;
      char c = data & 0xff;
      auto& ss_buf = print_bufs_[tid];
      ss_buf << c;
      if (c == '\n') {
        std::cout << std::dec << "#" << tid << ": " << ss_buf.str() << std::flush;
        ss_buf.str("");
      }         
    }
  }
  vortex_->io_req_ready = 1;
  vortex_->io_rsp_valid = 0;*/
}

void Simulator::eval_csr_bus() {
  if (csr_req_active_) { 
    if (vortex_->csr_req_valid && csr_req_ready_) {
    #ifndef NDEBUG
      if (vortex_->csr_req_rw)
        std::cout << std::dec << timestamp << ": [sim] CSR Wr Req: core=" << (int)vortex_->csr_req_coreid << ", addr=" << std::hex << vortex_->csr_req_addr << ", value=" << vortex_->csr_req_data << std::endl;
      else
        std::cout << std::dec << timestamp << ": [sim] CSR Rd Req: core=" << (int)vortex_->csr_req_coreid << ", addr=" << std::hex << vortex_->csr_req_addr << std::endl;
    #endif
      vortex_->csr_req_valid = 0;
      if (vortex_->csr_req_rw)
        csr_req_active_ = false;      
    }
    if (vortex_->csr_rsp_valid && vortex_->csr_rsp_ready) {
      *csr_rsp_value_ = vortex_->csr_rsp_data;
      vortex_->csr_rsp_ready = 0;
      csr_req_active_ = false;
    #ifndef NDEBUG
      std::cout << std::dec << timestamp << ": [sim] CSR Rsp: value=" << vortex_->csr_rsp_data << std::endl;
    #endif
    }
  } else {
    vortex_->csr_req_valid = 0;
    vortex_->csr_rsp_ready = 0;
  }
}

void Simulator::wait(uint32_t cycles) {
  for (int i = 0; i < cycles; ++i) {
    this->step();
  }
}

bool Simulator::is_busy() const {
  return vortex_->busy;
}

bool Simulator::csr_req_active() const {
  return csr_req_active_;
}

void Simulator::set_csr(int core_id, int addr, unsigned value) {
  vortex_->csr_req_valid  = 1;
  vortex_->csr_req_coreid = core_id;
  vortex_->csr_req_addr   = addr;
  vortex_->csr_req_rw     = 1;
  vortex_->csr_req_data   = value;  
  vortex_->csr_rsp_ready  = 0;

  csr_req_active_ = true;
}

void Simulator::get_csr(int core_id, int addr, unsigned *value) {
  vortex_->csr_req_valid  = 1;
  vortex_->csr_req_coreid = core_id;
  vortex_->csr_req_addr   = addr;
  vortex_->csr_req_rw     = 0;
  vortex_->csr_rsp_ready  = 1;

  csr_rsp_value_ = value;

  csr_req_active_ = true;  
}

void Simulator::run() {
#ifndef NDEBUG
  std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
#endif

  // execute program
  while (vortex_->busy 
      && !vortex_->ebreak) {
    this->step();
  }

  // wait 5 cycles to flush the pipeline
  this->wait(5);  
}

int Simulator::get_last_wb_value(int reg) const {
  return (int)vortex_->Vortex->genblk1__BRA__0__KET____DOT__cluster->genblk1__BRA__0__KET____DOT__core->pipeline->commit->writeback->last_wb_value[reg];
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