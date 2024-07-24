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

#include "cachesim.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <bitset>

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

double sc_time_stamp() {
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

CacheSim::CacheSim() {
  // create RTL module instance
  cache_ = new VVX_cache_top();

#ifdef VCD_OUTPUT
  Verilated::traceEverOn(true);
  trace_ = new VerilatedVcdC;
  cache_->trace(trace_, 99);
  trace_->open("trace.vcd");
#endif

  // force random values for uninitialized signals
  Verilated::randReset(2);

  ram_ = nullptr;
  mem_rsp_active_ = false;
  snp_req_active_ = false;
}

CacheSim::~CacheSim() {
#ifdef VCD_OUTPUT
  trace_->close();
#endif
  delete cache_;
  //need to delete the req and rsp vectors
}

void CacheSim::attach_ram(RAM* ram) {
  ram_ = ram;
  mem_rsp_vec_.clear();
}

void CacheSim::reset() {
#ifndef NDEBUG
  std::cout << timestamp << ": [sim] reset()" << std::endl;
#endif

  cache_->reset = 1;
  this->step();
  cache_->reset = 0;
  this->step();

  mem_rsp_vec_.clear();
  //clear req and rsp vecs

}

void CacheSim::step() {
  //std::cout << timestamp << ": [sim] step()" << std::endl;
  //toggle clock
  cache_->clk = 0;
  this->eval();

  cache_->clk = 1;
  this->eval();

  //handle core and memory reqs and rsps
  this->eval_reqs();
  this->eval_rsps();
  this->eval_mem_bus();
  timestamp++;
}

void CacheSim::eval() {
  cache_->eval();
#ifdef VCD_OUTPUT
  trace_->dump(timestamp);
#endif
  ++timestamp;
}

void CacheSim::run(){
//#ifndef NDEBUG

//#endif
  this->step();

  int valid = 300;
  int stalls = 20 + 10;

  while (valid > -1) {

      this->step();
      display_miss();
      if(cache_->core_rsp_valid){
        get_core_rsp();
      }

      if(!cache_->core_req_valid && !cache_->core_rsp_valid){
        valid--;

      }
      stalls--;
      if (stalls == 20){
          //stall_mem();
          //send_snoop_req();
          stalls--;
      }
  }
}

void CacheSim::clear_req(){
  cache_->core_req_valid = 0;
}

void CacheSim::send_req(core_req_t *req){
  core_req_vec_.push(req);
  unsigned int *data = new unsigned int[4];
  core_rsp_vec_.insert(std::pair<unsigned int, unsigned int*>(req->tag, data));
}

bool CacheSim::get_core_req_ready(){
  return cache_->core_req_ready;
}

bool CacheSim::get_core_rsp_ready(){
  return cache_->core_rsp_ready;
}

void CacheSim::eval_reqs(){
  //check to see if cache is accepting reqs
  if(!core_req_vec_.empty() && cache_->core_req_ready){
    core_req_t *req = core_req_vec_.front();

    cache_->core_req_valid = req->valid;
    cache_->core_req_rw = req->rw;
    cache_->core_req_byteen = req->byteen;

    cache_->core_req_addr[0] = req->addr[0];
    cache_->core_req_addr[1] = req->addr[1];
    cache_->core_req_addr[2] = req->addr[2];
    cache_->core_req_addr[3] = req->addr[3];

    cache_->core_req_data[0] = req->data[0];
    cache_->core_req_data[1] = req->data[1];
    cache_->core_req_data[2] = req->data[2];
    cache_->core_req_data[3] = req->data[3];

    cache_->core_req_tag = req->tag;

    core_req_vec_.pop();

  } else {
    clear_req();
  }
}

void CacheSim::eval_rsps(){
  //check to see if a request has been responded to
  if (cache_->core_rsp_valid){
      core_rsp_vec_.at(cache_->core_rsp_tag)[0] = cache_->core_rsp_data[0];
      core_rsp_vec_.at(cache_->core_rsp_tag)[1] = cache_->core_rsp_data[1];
      core_rsp_vec_.at(cache_->core_rsp_tag)[2] = cache_->core_rsp_data[2];
      core_rsp_vec_.at(cache_->core_rsp_tag)[3] = cache_->core_rsp_data[3];
  }
}

void CacheSim::stall_mem(){
  cache_->mem_req_ready = 0;
}

void CacheSim::send_snoop_req(){
    /*cache_->snp_req_valid = 1;
    cache_->snp_req_addr = 0x12222222;
    cache_->snp_req_invalidate = 1;
    cache_->snp_req_tag = 0xff; */
}

void CacheSim::eval_mem_bus() {
  if (ram_ == nullptr) {
    cache_->mem_req_ready = 0;
    return;
  }

  // schedule memory responses
  int dequeue_index = -1;
  for (int i = 0; i < mem_rsp_vec_.size(); i++) {
    if (mem_rsp_vec_[i].cycles_left > 0) {
      mem_rsp_vec_[i].cycles_left -= 1;
    }
    if ((dequeue_index == -1)
     && (mem_rsp_vec_[i].cycles_left == 0)) {
      dequeue_index = i;
    }
  }

  // send memory response
  if (mem_rsp_active_
   && cache_->mem_rsp_valid
   && cache_->mem_rsp_ready) {
    mem_rsp_active_ = false;
  }
  if (!mem_rsp_active_) {
    if (dequeue_index != -1) { //time to respond to the request
      cache_->mem_rsp_valid = 1;

      //copy data from the rsp queue to the cache module
      memcpy(cache_->mem_rsp_data.data(), mem_rsp_vec_[dequeue_index].data, MEM_BLOCK_SIZE);

      cache_->mem_rsp_tag = mem_rsp_vec_[dequeue_index].tag;
      free(mem_rsp_vec_[dequeue_index].data); //take data out of the queue
      mem_rsp_vec_.erase(mem_rsp_vec_.begin() + dequeue_index);
      mem_rsp_active_ = true;
    } else {
      cache_->mem_rsp_valid = 0;
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
    if (cache_->mem_req_valid) {
      if (cache_->mem_req_rw) { //write = 1
        uint64_t byteen = cache_->mem_req_byteen;
        uint64_t base_addr = (cache_->mem_req_addr * MEM_BLOCK_SIZE);
        uint8_t* data = reinterpret_cast<uint8_t*>(cache_->mem_req_data.data());
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          if ((byteen >> i) & 0x1) {
            (*ram_)[base_addr + i] = data[i];
          }
        }
      } else {
        mem_req_t mem_req;
        mem_req.cycles_left = MEM_LATENCY;
        mem_req.data = (uint8_t*)malloc(MEM_BLOCK_SIZE);
        mem_req.tag = cache_->mem_req_tag;
        ram_->read(cache_->mem_req_addr * MEM_BLOCK_SIZE, MEM_BLOCK_SIZE, mem_req.data);
        mem_rsp_vec_.push_back(mem_req);
      }
    }
  }

  cache_->mem_req_ready = ~mem_stalled;
}

bool CacheSim::assert_equal(unsigned int* data, unsigned int tag){
  int check = 0;
  unsigned int *rsp = core_rsp_vec_.at(tag);
  for (int i = 0; i < 4; ++i){
    for (int j = 0; j < 4; ++j){
      if (data[i] == rsp[j]){
        check++;
      }
    }
  }

  return check;

}

//DEBUG

void CacheSim::display_miss(){
  //int i = (unsigned int)cache_->miss_vec;
  //std::bitset<8> x(i);
  //if (i) std::cout << "Miss Vec " << x << std::endl;
  //std::cout << "Miss Vec 0" << cache_->miss_vec[0] << std::endl;
}

void CacheSim::get_core_req(unsigned int (&rsp)[4]){
  rsp[0] = cache_->core_rsp_data[0];
  rsp[1] = cache_->core_rsp_data[1];
  rsp[2] = cache_->core_rsp_data[2];
  rsp[3] = cache_->core_rsp_data[3];

  //std::cout << std::hex << "core_rsp_valid: " << cache_->core_rsp_valid << std::endl;
  //std::cout << std::hex << "core_rsp_data: " << cache_->core_rsp_data << std::endl;
  //std::cout << std::hex << "core_rsp_tag: " << cache_->core_rsp_tag << std::endl;
}

void CacheSim::get_core_rsp(){
  //std::cout << cache_->genblk5_BRA_0_KET_->bank->is_fill_in_pipe<< std::endl;
  char check = cache_->core_rsp_valid;
  std::cout << std::hex << "core_rsp_valid: " << (unsigned int) check << std::endl;
  std::cout << std::hex << "core_rsp_data[0]: " << cache_->core_rsp_data[0] << std::endl;
  std::cout << std::hex << "core_rsp_data[1]: " << cache_->core_rsp_data[1] << std::endl;
  std::cout << std::hex << "core_rsp_data[2]: " << cache_->core_rsp_data[2] << std::endl;
  std::cout << std::hex << "core_rsp_data[3]: " << cache_->core_rsp_data[3] << std::endl;
  std::cout << std::hex << "core_rsp_tag: " << cache_->core_rsp_tag << std::endl;
}

void CacheSim::get_mem_req(){
  std::cout << std::hex << "mem_req_valid: " << cache_->mem_req_valid << std::endl;
  std::cout << std::hex << "mem_req_rw: " << cache_->mem_req_rw << std::endl;
  std::cout << std::hex << "mem_req_byteen: " << cache_->mem_req_byteen << std::endl;
  std::cout << std::hex << "mem_req_addr: " << cache_->mem_req_addr << std::endl;
  std::cout << std::hex << "mem_req_data: " << cache_->mem_req_data << std::endl;
  std::cout << std::hex << "mem_req_tag: " << cache_->mem_req_tag << std::endl;
}

void CacheSim::get_mem_rsp(){
  std::cout << std::hex << "mem_rsp_valid: " << cache_->mem_rsp_valid << std::endl;
  std::cout << std::hex << "mem_rsp_data: " << cache_->mem_rsp_data << std::endl;
  std::cout << std::hex << "mem_rsp_tag: " << cache_->mem_rsp_tag << std::endl;
  std::cout << std::hex << "mem_rsp_ready: " << cache_->mem_rsp_ready << std::endl;
}
