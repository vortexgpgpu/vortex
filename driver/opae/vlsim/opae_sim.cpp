#include "opae_sim.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define CCI_LATENCY  8
#define CCI_RAND_MOD 8
#define CCI_RQ_SIZE 16
#define CCI_WQ_SIZE 16

#define RESET_DELAY 4

#define ENABLE_DRAM_STALLS
#define DRAM_LATENCY 24
#define DRAM_RQ_SIZE 16
#define DRAM_STALLS_MODULO 16

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

opae_sim::opae_sim() {  
  // force random values for unitialized signals  
  Verilated::randReset(2);
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
  stop_ = false;
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
  delete vortex_afu_;
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

int opae_sim::prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
  auto alloc = __aligned_malloc(CACHE_BLOCK_SIZE, len);
  if (alloc == NULL)
    return -1;
  host_buffer_t buffer;
  buffer.data   = (uint64_t*)alloc;
  buffer.size   = len;
  buffer.ioaddr = uintptr_t(alloc); 
  auto index = host_buffers_.size();
  host_buffers_.emplace(index, buffer);
  *buf_addr = alloc;
  *wsid = index;
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

void opae_sim::flush() {
  // flush pending CCI requests  
}

///////////////////////////////////////////////////////////////////////////////

void opae_sim::reset() {
  
  host_buffers_.clear();
  dram_reads_.clear();
  cci_reads_.clear();
  cci_writes_.clear();
  vortex_afu_->vcp2af_sRxPort_c0_rspValid = 0;  
  vortex_afu_->vcp2af_sRxPort_c1_rspValid = 0;  
  vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull = 0;
  vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull = 0;
  vortex_afu_->avs_readdatavalid = 0;  
  vortex_afu_->avs_waitrequest = 0;

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
  // update DRAM responses schedule
  for (auto& rsp : dram_reads_) {
    if (rsp.cycles_left > 0)
      rsp.cycles_left -= 1;
  }

  // schedule DRAM responses in FIFO order
  std::list<dram_rd_req_t>::iterator dram_rd_it(dram_reads_.end());
  if (!dram_reads_.empty() 
   && (0 == dram_reads_.begin()->cycles_left)) {
      dram_rd_it = dram_reads_.begin();
  }

  // send DRAM response  
  vortex_afu_->avs_readdatavalid = 0;  
  if (dram_rd_it != dram_reads_.end()) {
    vortex_afu_->avs_readdatavalid = 1;
    memcpy(vortex_afu_->avs_readdata, dram_rd_it->data.data(), DRAM_BLOCK_SIZE);
    uint32_t addr = dram_rd_it->addr;
    dram_reads_.erase(dram_rd_it);
    /*printf("%0ld: [sim] DRAM Rd Rsp: addr=%x, pending={", timestamp, addr * DRAM_BLOCK_SIZE);
    for (auto& req : dram_reads_) {
      if (req.cycles_left != 0) 
        printf(" !%0x", req.addr * DRAM_BLOCK_SIZE);
      else
        printf(" %0x", req.addr * DRAM_BLOCK_SIZE);
    }
    printf("}\n");*/
  }

  // handle DRAM stalls
  bool dram_stalled = false;
#ifdef ENABLE_DRAM_STALLS
  if (0 == ((timestamp/2) % DRAM_STALLS_MODULO)) { 
    dram_stalled = true;
  } else
  if (dram_reads_.size() >= DRAM_RQ_SIZE) {
    dram_stalled = true;
  }
#endif

  // process DRAM requests
  if (!dram_stalled) {
    assert(!vortex_afu_->avs_read || !vortex_afu_->avs_write);
    if (vortex_afu_->avs_write) {           
      uint64_t byteen = vortex_afu_->avs_byteenable;
      unsigned base_addr = vortex_afu_->avs_address * DRAM_BLOCK_SIZE;
      uint8_t* data = (uint8_t*)(vortex_afu_->avs_writedata);
      for (int i = 0; i < DRAM_BLOCK_SIZE; i++) {
        if ((byteen >> i) & 0x1) {            
          ram_[base_addr + i] = data[i];
        }
      }
      /*printf("%0ld: [sim] DRAM Wr Req: addr=%x, data=", timestamp, base_addr);
      for (int i = 0; i < DRAM_BLOCK_SIZE; i++) {
        printf("%0x", data[(DRAM_BLOCK_SIZE-1)-i]);
      }
      printf("\n");*/
    }
    if (vortex_afu_->avs_read) {
      dram_rd_req_t dram_req;      
      dram_req.addr = vortex_afu_->avs_address;
      ram_.read(vortex_afu_->avs_address * DRAM_BLOCK_SIZE, DRAM_BLOCK_SIZE, dram_req.data.data());      
      dram_req.cycles_left = DRAM_LATENCY;
      for (auto& rsp : dram_reads_) {
        if (dram_req.addr == rsp.addr) {
          dram_req.cycles_left = rsp.cycles_left;
          break;
        }
      }
      dram_reads_.emplace_back(dram_req);
      /*printf("%0ld: [sim] DRAM Rd Req: addr=%x, pending={", timestamp, dram_req.addr * DRAM_BLOCK_SIZE);
      for (auto& req : dram_reads_) {
        if (req.cycles_left != 0) 
          printf(" !%0x", req.addr * DRAM_BLOCK_SIZE);
        else
          printf(" %0x", req.addr * DRAM_BLOCK_SIZE);
      }
      printf("}\n");*/
    }
  }

  vortex_afu_->avs_waitrequest = dram_stalled;
}