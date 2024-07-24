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

#include <random>
#include "memsim.h"
#include "ram.h"

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static bool trace_enabled = false;
static uint64_t trace_start_time = 0;
static uint64_t trace_stop_time = -1ull;
static uint64_t timestamp = 0;

double sc_time_stamp() {
  	return timestamp;
}

bool sim_trace_enabled() {
	if (timestamp >= trace_start_time
	&& timestamp < trace_stop_time)
			return true;
	return trace_enabled;
}

void sim_trace_enable (bool enable) {
  	trace_enabled = enable;
}

int generate_rand (int min, int max) {
	int range = max - min + 1;
	return rand() % range + min;
}

int generate_rand_mask (int mask) {
	int result = 0;
	int m = mask;
	for (int i = 0; i < 4; i++) {
		int bit = m & 0b1;
		int rand_bit = generate_rand (0, bit);
		result |= (rand_bit << i);
		m = m >> 1;
	}
	return result;
}

MemSim::MemSim() {
	// create RTL module instance
	msu_ = new VVX_mem_scheduler();

#ifdef VCD_OUTPUT
  	Verilated::traceEverOn(true);
  	trace_ = new VerilatedVcdC;
  	cache_->trace(trace_, 99);
  	race_->open("trace.vcd");
#endif

  // force random values for uninitialized signals
  Verilated::randReset(2);
}

MemSim::~MemSim() {
#ifdef VCD_OUTPUT
	trace_->close();
#endif
	delete msu_;
}

void MemSim::eval() {
	msu_->eval();
#ifdef VCD_OUTPUT
	trace_->dump(timestamp++);
#endif
}

void MemSim::step() {
	msu_->clk = 0;
	this->eval();

	msu_->clk = 1;
	this->eval();
}

void MemSim::reset() {
	msu_->reset = 1;
	this->step();

	msu_->reset = 0;
	this->step();
}

void MemSim::attach_core() {
	if (msu_->core_req_ready) {
		msu_->core_req_valid 	= generate_rand(0, 1);
		msu_->core_req_rw 		= generate_rand(0, 1);
		msu_->core_req_mask 		= generate_rand(0b0001, 0b1111);
		msu_->core_req_byteen 	= 0b1;
		msu_->core_req_addr 		= generate_rand(0, 0x10000000);
		msu_->core_req_data 		= generate_rand(0x60000000, 0x80000000);
		msu_->core_req_tag 		= generate_rand(0x00, 0xFF);
	}
	msu_->core_rsp_ready = true;
}

void MemSim::attach_ram (RAM *ram) {

	req_t req;
	req.valid 		= msu_->mem_req_valid;
	req.rw 				= msu_->mem_req_rw;
	req.byteen		= msu_->mem_req_byteen;
	req.addr 			= msu_->mem_req_addr;
	req.data 			= msu_->mem_req_data;
	req.tag 			= msu_->mem_req_tag;
	msu_->mem_req_ready = ram->is_ready();

	ram->insert_req(req);

	rsp_t rsp;
	rsp = ram->schedule_rsp();

	msu_->mem_rsp_valid = rsp.valid;
	msu_->mem_rsp_data 	= rsp.data;
	msu_->mem_rsp_tag 	= rsp.tag;
	rsp.ready 			= msu_->mem_rsp_ready;
	std::cout<<"MEMSIM: mem_rsp_ready: "<<rsp.ready<<"\n";

	ram->halt_rsp(rsp);
}

void MemSim::run(RAM *ram) {
	this->reset();

	while (sc_time_stamp() < SIM_TIME) {
		this->step();
		std::cout<<"========================="<<"\n";
		std::cout<<"Cycle: "<<sc_time_stamp()<<"\n";
		this->attach_core();
		this->attach_ram(ram);
	}
}

int main (int argc, char** argv, char** env) {
    Verilated::commandArgs(argc, argv);

	MemSim memsim;
	RAM ram;

	memsim.run(&ram);

	return 0;
}
