#include "opae_sim.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define CCI_LATENCY 8
#define CCI_RAND_MOD 8
#define CCI_RQ_SIZE 16
#define CCI_WQ_SIZE 16

#define ENABLE_DRAM_STALLS
#define DRAM_LATENCY 4
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

  this->reset();

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

int opae_sim::prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
  auto alloc = aligned_alloc(CACHE_BLOCK_SIZE, len);
  if (alloc == NULL)
    return -1;
  host_buffer_t buffer;
  buffer.data   = (uint64_t*)alloc;
  buffer.size   = len;
  buffer.ioaddr = intptr_t(alloc) / CACHE_BLOCK_SIZE; 
  auto index = host_buffers_.size();
  host_buffers_.push_back(buffer);
  *buf_addr = alloc;
  *wsid = index;
  return 0;
}

void opae_sim::release_buffer(uint64_t wsid) {
  free(host_buffers_[wsid].data);
  host_buffers_.erase(host_buffers_.begin() + wsid);
}

void opae_sim::get_io_address(uint64_t wsid, uint64_t *ioaddr) {
  *ioaddr = host_buffers_[wsid].ioaddr * CACHE_BLOCK_SIZE;
}

void opae_sim::write_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t value) {
  std::lock_guard<std::mutex> guard(mutex_);
  
  vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid = 1;  
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
  memcpy(vortex_afu_->vcp2af_sRxPort_c0_data, &value, 8);
  this->step();
  assert(!vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid);
}

void opae_sim::read_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t *value) {
  std::lock_guard<std::mutex> guard(mutex_);

  vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
  vortex_afu_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
  this->step();  
  assert(!vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid);
  assert(vortex_afu_->af2cp_sTxPort_c2_mmioRdValid);  
  *value = vortex_afu_->af2cp_sTxPort_c2_data;
}

void opae_sim::flush() {
  // flush pending CCI requests  
}

///////////////////////////////////////////////////////////////////////////////

void opae_sim::reset() {
  vortex_afu_->reset = 1;
  this->step();  
  vortex_afu_->reset = 0;

  // Turn on assertion after reset
  Verilated::assertOn(true);
}

void opae_sim::step() {
  vortex_afu_->clk = 0;
  this->eval();

  vortex_afu_->clk = 1;
  this->eval();

  this->sRxPort_bus();
  this->sTxPort_bus();
  this->avs_bus();
}

void opae_sim::eval() {  
  vortex_afu_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void opae_sim::sRxPort_bus() {
  // schedule CCI read responses
  int cci_rd_index = -1;
  for (int i = 0; i < cci_reads_.size(); i++) {
    if (cci_reads_[i].cycles_left > 0) {
      cci_reads_[i].cycles_left -= 1;
    }
    if ((cci_rd_index == -1) 
     && (cci_reads_[i].cycles_left == 0)) {
      cci_rd_index = i;
    }
  }

  // schedule CCI write responses
  int cci_wr_index = -1;
  for (int i = 0; i < cci_writes_.size(); i++) {
    if (cci_writes_[i].cycles_left > 0) {
      cci_writes_[i].cycles_left -= 1;
    }
    if ((cci_wr_index == -1) 
     && (cci_writes_[i].cycles_left == 0)) {
      cci_wr_index = i;
    }
  }

  // send CCI read response  
  vortex_afu_->vcp2af_sRxPort_c0_rspValid = 0;  
  if (cci_rd_index != -1) {
    vortex_afu_->vcp2af_sRxPort_c0_rspValid = 1;
    memcpy(vortex_afu_->vcp2af_sRxPort_c0_data, cci_reads_[cci_rd_index].block.data(), CACHE_BLOCK_SIZE);
    vortex_afu_->vcp2af_sRxPort_c0_hdr_mdata = cci_reads_[cci_rd_index].mdata;
    cci_reads_.erase(cci_reads_.begin() + cci_rd_index);
  }

  // send CCI write response  
  vortex_afu_->vcp2af_sRxPort_c1_rspValid = 0;  
  if (cci_wr_index != -1) {
    vortex_afu_->vcp2af_sRxPort_c1_rspValid = 1;
    vortex_afu_->vcp2af_sRxPort_c1_hdr_mdata = cci_writes_[cci_wr_index].mdata;
    cci_writes_.erase(cci_writes_.begin() + cci_wr_index);
  }

  // mmio
  vortex_afu_->vcp2af_sRxPort_c0_mmioWrValid = 0;
  vortex_afu_->vcp2af_sRxPort_c0_mmioRdValid = 0;
}
  
void opae_sim::sTxPort_bus() {
  // check read queue size
  vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull = (cci_reads_.size() >= CCI_RQ_SIZE);

  // check write queue size
  vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull = (cci_writes_.size() >= CCI_WQ_SIZE);

  // process read requests
  if (vortex_afu_->af2cp_sTxPort_c0_valid && !vortex_afu_->vcp2af_sRxPort_c0_TxAlmFull) {
    cci_rd_req_t cci_req;
    cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);     
    cci_req.mdata = vortex_afu_->af2cp_sTxPort_c0_hdr_mdata;
    auto host_ptr = this->to_host_ptr(vortex_afu_->af2cp_sTxPort_c0_hdr_address);
    memcpy(cci_req.block.data(), host_ptr, CACHE_BLOCK_SIZE);
    cci_reads_.push_back(cci_req);
  }

  // process write requests
  if (vortex_afu_->af2cp_sTxPort_c1_valid && !vortex_afu_->vcp2af_sRxPort_c1_TxAlmFull) {
    cci_wr_req_t cci_req;
    cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);
    cci_req.mdata = vortex_afu_->af2cp_sTxPort_c1_hdr_mdata;
    auto host_ptr = this->to_host_ptr(vortex_afu_->af2cp_sTxPort_c1_hdr_address);
    memcpy(host_ptr, vortex_afu_->af2cp_sTxPort_c1_data, CACHE_BLOCK_SIZE);
    cci_writes_.push_back(cci_req);
  } 
}
  
void opae_sim::avs_bus() {
  // schedule DRAM read responses
  int dram_rd_index = -1;
  for (int i = 0; i < dram_reads_.size(); i++) {
    if (dram_reads_[i].cycles_left > 0) {
      dram_reads_[i].cycles_left -= 1;
    }
    if ((dram_rd_index == -1) 
     && (dram_reads_[i].cycles_left == 0)) {
      dram_rd_index = i;
    }
  }

  // send DRAM response  
  vortex_afu_->avs_readdatavalid = 0;  
  if (dram_rd_index != -1) {
    vortex_afu_->avs_readdatavalid = 1;
    memcpy(vortex_afu_->avs_readdata, dram_reads_[dram_rd_index].block.data(), CACHE_BLOCK_SIZE);
    dram_reads_.erase(dram_reads_.begin() + dram_rd_index);
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
    if (vortex_afu_->avs_write) {
      assert(0 == vortex_afu_->mem_bank_select);
      uint64_t byteen = vortex_afu_->avs_byteenable;
      unsigned base_addr = (vortex_afu_->avs_address * CACHE_BLOCK_SIZE);
      uint8_t* data = (uint8_t*)(vortex_afu_->avs_writedata);
      for (int i = 0; i < CACHE_BLOCK_SIZE; i++) {
        if ((byteen >> i) & 0x1) {            
          ram_[base_addr + i] = data[i];
        }
      }
    }
    if (vortex_afu_->avs_read) {
      assert(0 == vortex_afu_->mem_bank_select);
      dram_rd_req_t dram_req;
      dram_req.cycles_left = DRAM_LATENCY;     
      unsigned base_addr = (vortex_afu_->avs_address * CACHE_BLOCK_SIZE);
      ram_.read(base_addr, CACHE_BLOCK_SIZE, dram_req.block.data());
      dram_reads_.push_back(dram_req);
    }   
  }

  vortex_afu_->avs_waitrequest = dram_stalled;
}

uint64_t* opae_sim::to_host_ptr(uint64_t ioaddr) {
  for (auto& buffer : host_buffers_) {
    if (ioaddr >= buffer.ioaddr 
     && ioaddr < (buffer.ioaddr + buffer.size)) {
      return buffer.data + (ioaddr - buffer.ioaddr) * (CACHE_BLOCK_SIZE / 8);
    }
  }
  printf("error: to_host_ptr(0x%lx) failed\n", ioaddr);
  std::abort();
  return nullptr;
}