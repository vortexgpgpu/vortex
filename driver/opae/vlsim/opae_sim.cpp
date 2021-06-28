#include "opae_sim.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define CCI_LATENCY  8
#define CCI_RAND_MOD 8
#define CCI_RQ_SIZE 16
#define CCI_WQ_SIZE 16

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

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

static void *__aligned_malloc(size_t alignment, size_t size) {
  // reserve margin for alignment and storing of unaligned address
  size_t margin = (alignment-1) + sizeof(void*);
  void *unaligned_addr = malloc(size + margin);
  void **aligned_addr = (void**)((uintptr_t)(((uint8_t*)unaligned_addr) + margin) & ~(alignment-1));
  aligned_addr[-1] = unaligned_addr;
  return aligned_addr;
}

static void __aligned_free(void *ptr) {
  // retreive the stored unaligned address and use it to free the allocation
  void* unaligned_addr = ((void**)ptr)[-1];
  free(unaligned_addr);
}

///////////////////////////////////////////////////////////////////////////////

opae_sim::opae_sim() 
  : stop_(false)
  , host_buffer_ids_(0) 
{  
  // force random values for unitialized signals  
  Verilated::randReset(VERILATOR_RESET_VALUE);
  Verilated::randSeed(50);

  // Turn off assertion before reset
  Verilated::assertOn(false);

  vortex_afu_ = new Vvortex_afu_shim();

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC();
  vortex_afu_->trace(trace_, 99);
  trace_->open("trace.vcd");
#endif

  // reset the device
  this->reset();

  // launch execution thread
  future_ = std::async(std::launch::async, [&]{                   
      while (!stop_) {
          std::lock_guard<std::mutex> guard(mutex_);
          this->step();
      }
  }); 
}

opae_sim::~opae_sim() {  
  stop_ = true;
  if (future_.valid()) {
    future_.wait();
  }
#ifdef VCD_OUTPUT
  trace_->close();
#endif  
  for (auto& buffer : host_buffers_) {
    __aligned_free(buffer.second.data);
  }   
  delete vortex_afu_;
}

int opae_sim::prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
  auto alloc = __aligned_malloc(CACHE_BLOCK_SIZE, len);
  if (alloc == NULL)
    return -1;
  host_buffer_t buffer;
  buffer.data   = (uint64_t*)alloc;
  buffer.size   = len;
  buffer.ioaddr = uintptr_t(alloc); 
  auto buffer_id = host_buffer_ids_++;
  host_buffers_.emplace(buffer_id, buffer);
  *buf_addr = alloc;
  *wsid = buffer_id;
  return 0;
}

void opae_sim::release_buffer(uint64_t wsid) {
  auto it = host_buffers_.find(wsid);
  if (it != host_buffers_.end()) {
    __aligned_free(it->second.data);
    host_buffers_.erase(it);
  }
}

void opae_sim::get_io_address(uint64_t wsid, uint64_t *ioaddr) {
  *ioaddr = host_buffers_[wsid].ioaddr;
}

void opae_sim::read_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t *value) {
  std::lock_guard<std::mutex> guard(mutex_);

  vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
  this->step();  
  vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid = 0;
  assert(vortex_afu_->af2cp_sTxPort_c2_mmioRdValid);  
  *value = vortex_afu_->af2cp_sTxPort_c2_data;
}

void opae_sim::write_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t value) {
  std::lock_guard<std::mutex> guard(mutex_);
  
  vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid = 1;  
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
  memcpy(vortex_afu_->vcp2af_sRxPort_c0_data, &value, 8);
  this->step();
  vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid = 0;
}

///////////////////////////////////////////////////////////////////////////////

void opae_sim::reset() {  
  cci_reads_.clear();
  cci_writes_.clear();
  vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid = 0;
  vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid = 0;
  vortex_afu_->vcp2af_sRxPort_c0_rspValid = 0;  
  vortex_afu_->vcp2af_sRxPort_c1_rspValid = 0;  
  vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull = 0;
  vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull = 0;

  for (int b = 0; b < MEMORY_BANKS; ++b) {
    mem_reads_[b].clear();
    vortex_afu_->avs_readdatavalid[b] = 0;  
    vortex_afu_->avs_waitrequest[b] = 0;
  }

  vortex_afu_->reset = 1;

  for (int i = 0; i < RESET_DELAY; ++i) {
    vortex_afu_->clk = 0;
    this->eval();
    vortex_afu_->clk = 1;
    this->eval();
  }  

  vortex_afu_->reset = 0;
  
  // Turn on assertion after reset
  Verilated::assertOn(true);
}

void opae_sim::step() {
  this->sRxPort_bus();
  this->sTxPort_bus();
  this->avs_bus();
  
  vortex_afu_->clk = 0;
  this->eval();
  vortex_afu_->clk = 1;
  this->eval();

#ifndef NDEBUG
  fflush(stdout);
#endif
}

void opae_sim::eval() {  
  vortex_afu_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void opae_sim::sRxPort_bus() {      
  // check mmio request
  bool mmio_req_enabled = vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid
                       || vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid;

  // schedule CCI read responses
  std::list<cci_rd_req_t>::iterator cci_rd_it(cci_reads_.end());
  for (auto it = cci_reads_.begin(), ie = cci_reads_.end(); it != ie; ++it) {
    if (it->cycles_left > 0)
      it->cycles_left -= 1;
    if ((cci_rd_it == ie) && (it->cycles_left == 0)) {
      cci_rd_it = it;
    }
  }

  // schedule CCI write responses
  std::list<cci_wr_req_t>::iterator cci_wr_it(cci_writes_.end());
  for (auto it = cci_writes_.begin(), ie = cci_writes_.end(); it != ie; ++it) {
    if (it->cycles_left > 0)
      it->cycles_left -= 1;
    if ((cci_wr_it == ie) && (it->cycles_left == 0)) {
      cci_wr_it = it;
    }
  }

  // send CCI write response  
  vortex_afu_->vcp2af_sRxPort_c1_rspValid = 0;  
  if (cci_wr_it != cci_writes_.end()) {
    vortex_afu_->vcp2af_sRxPort_c1_rspValid = 1;
    vortex_afu_->vcp2af_sRxPort_c1_hdr_resp_type = 0;
    vortex_afu_->vcp2af_sRxPort_c1_hdr_mdata = cci_wr_it->mdata;
    cci_writes_.erase(cci_wr_it);
  }

  // send CCI read response (ensure mmio disabled) 
  vortex_afu_->vcp2af_sRxPort_c0_rspValid = 0;  
  if (!mmio_req_enabled 
   && (cci_rd_it != cci_reads_.end())) {
    vortex_afu_->vcp2af_sRxPort_c0_rspValid = 1;
    vortex_afu_->vcp2af_sRxPort_c0_hdr_resp_type = 0;
    memcpy(vortex_afu_->vcp2af_sRxPort_c0_data, cci_rd_it->data.data(), CACHE_BLOCK_SIZE);
    vortex_afu_->vcp2af_sRxPort_c0_hdr_mdata = cci_rd_it->mdata;    
    /*printf("%0ld: [sim] CCI Rd Rsp: addr=%ld, mdata=%d, data=", timestamp, cci_rd_it->addr, cci_rd_it->mdata);
    for (int i = 0; i < CACHE_BLOCK_SIZE; ++i)
      printf("%02x", cci_rd_it->data[CACHE_BLOCK_SIZE-1-i]);
    printf("\n");*/
    cci_reads_.erase(cci_rd_it);
  }
}
  
void opae_sim::sTxPort_bus() {
  // process read requests
  if (vortex_afu_->af2cp_sTxPort_c0_valid) {
    assert(!vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull);
    cci_rd_req_t cci_req;
    cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);     
    cci_req.addr = vortex_afu_->af2cp_sTxPort_c0_hdr_address;
    cci_req.mdata = vortex_afu_->af2cp_sTxPort_c0_hdr_mdata;
    auto host_ptr = (uint64_t*)(vortex_afu_->af2cp_sTxPort_c0_hdr_address * CACHE_BLOCK_SIZE);
    memcpy(cci_req.data.data(), host_ptr, CACHE_BLOCK_SIZE);
    //printf("%0ld: [sim] CCI Rd Req: addr=%ld, mdata=%d\n", timestamp, vortex_afu_->af2cp_sTxPort_c0_hdr_address, cci_req.mdata);
    cci_reads_.emplace_back(cci_req);    
  }

  // process write requests
  if (vortex_afu_->af2cp_sTxPort_c1_valid) {
    assert(!vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull);
    cci_wr_req_t cci_req;
    cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);
    cci_req.mdata = vortex_afu_->af2cp_sTxPort_c1_hdr_mdata;
    auto host_ptr = (uint64_t*)(vortex_afu_->af2cp_sTxPort_c1_hdr_address * CACHE_BLOCK_SIZE);
    memcpy(host_ptr, vortex_afu_->af2cp_sTxPort_c1_data, CACHE_BLOCK_SIZE);
    cci_writes_.emplace_back(cci_req);
  } 

  // check queues overflow
  vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull = (cci_reads_.size() >= (CCI_RQ_SIZE-1));
  vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull = (cci_writes_.size() >= (CCI_WQ_SIZE-1));
}
  
void opae_sim::avs_bus() {
  for (int b = 0; b < MEMORY_BANKS; ++b) {
    // update memory responses schedule
    for (auto& rsp : mem_reads_[b]) {
      if (rsp.cycles_left > 0)
        rsp.cycles_left -= 1;
    }

    // schedule memory responses in FIFO order
    std::list<mem_rd_req_t>::iterator mem_rd_it(mem_reads_[b].end());
    if (!mem_reads_[b].empty() 
    && (0 == mem_reads_[b].begin()->cycles_left)) {
        mem_rd_it = mem_reads_[b].begin();
    }

    // send memory response  
    vortex_afu_->avs_readdatavalid[b] = 0;  
    if (mem_rd_it != mem_reads_[b].end()) {
      vortex_afu_->avs_readdatavalid[b] = 1;
      memcpy(vortex_afu_->avs_readdata[b], mem_rd_it->data.data(), MEM_BLOCK_SIZE);
      uint32_t addr = mem_rd_it->addr;
      mem_reads_[b].erase(mem_rd_it);
      /*printf("%0ld: [sim] MEM Rd Rsp: bank=%d, addr=%x, pending={", timestamp, b, addr * MEM_BLOCK_SIZE);
      for (auto& req : mem_reads_[b]) {
        if (req.cycles_left != 0) 
          printf(" !%0x", req.addr * MEM_BLOCK_SIZE);
        else
          printf(" %0x", req.addr * MEM_BLOCK_SIZE);
      }
      printf("}\n");*/
    }

    // handle memory stalls
    bool mem_stalled = false;
  #ifdef ENABLE_MEM_STALLS
    if (0 == ((timestamp/2) % MEM_STALLS_MODULO)) { 
      mem_stalled = true;
    } else
    if (mem_reads_[b].size() >= MEM_RQ_SIZE) {
      mem_stalled = true;
    }
  #endif

    // process memory requests
    if (!mem_stalled) {
      assert(!vortex_afu_->avs_read[b] || !vortex_afu_->avs_write[b]);
      if (vortex_afu_->avs_write[b]) {           
        uint64_t byteen = vortex_afu_->avs_byteenable[b];
        unsigned base_addr = vortex_afu_->avs_address[b] * MEM_BLOCK_SIZE;
        uint8_t* data = (uint8_t*)(vortex_afu_->avs_writedata[b]);
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          if ((byteen >> i) & 0x1) {            
            ram_[base_addr + i] = data[i];
          }
        }
        /*printf("%0ld: [sim] MEM Wr Req: bank=%d, addr=%x, data=", timestamp, b, base_addr);
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          printf("%0x", data[(MEM_BLOCK_SIZE-1)-i]);
        }
        printf("\n");*/
      }
      if (vortex_afu_->avs_read[b]) {
        mem_rd_req_t mem_req;      
        mem_req.addr = vortex_afu_->avs_address[b];
        ram_.read(vortex_afu_->avs_address[b] * MEM_BLOCK_SIZE, MEM_BLOCK_SIZE, mem_req.data.data());      
        mem_req.cycles_left = MEM_LATENCY;
        for (auto& rsp : mem_reads_[b]) {
          if (mem_req.addr == rsp.addr) {
            mem_req.cycles_left = rsp.cycles_left;
            break;
          }
        }
        mem_reads_[b].emplace_back(mem_req);
        /*printf("%0ld: [sim] MEM Rd Req: bank=%d, addr=%x, pending={", timestamp, b, mem_req.addr * MEM_BLOCK_SIZE);
        for (auto& req : mem_reads_[b]) {
          if (req.cycles_left != 0) 
            printf(" !%0x", req.addr * MEM_BLOCK_SIZE);
          else
            printf(" %0x", req.addr * MEM_BLOCK_SIZE);
        }
        printf("}\n");*/
      }
    }

    vortex_afu_->avs_waitrequest[b] = mem_stalled;
  }
}