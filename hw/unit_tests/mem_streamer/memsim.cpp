#include <iostream>
#include <verilated.h>
#include "memsim.h"

// Number of clock edges that we'll simulate
#define SIM_TIME 50

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

void generate_req(req_t *req) {
	req->valid = 1;
	req->rw = 0;
	req->byteen = 1;
	req->addr = 0x04030201;
	req->data = 0x12345678;
	req->tag = 0xAB;
}

MemSim::MemSim() {
	msu_ = new VVX_mem_streamer_test();

	// Enable tracing
	Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC;
    msu_->trace(trace_, 99);
    trace_->open("trace.vcd");
}

MemSim::~MemSim() {
	trace_->close();
	delete msu_;
}

void MemSim::eval() {
	msu_->eval();
	trace_->dump(timestamp++);
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

void MemSim::set_core_req(req_t *core_req) {
	msu_->req_valid 	= core_req->valid;
	msu_->req_rw 		= core_req->rw;
	msu_->req_mask 		= core_req->mask;
	msu_->req_byteen 	= core_req->byteen;
	msu_->req_addr 		= core_req->addr;
	msu_->req_data 		= core_req->data;
	msu_->req_tag 		= core_req->tag;
}

int main (int argc, char** argv, char** env) {
    Verilated::commandArgs(argc, argv);

	MemSim memsim;
	req_t input;

	generate_req(&input);

	memsim.step();
	memsim.reset();
	memsim.set_core_req(&input);
	memsim.step();
	memsim.step();
	memsim.step();
	memsim.step();
	memsim.step();
	memsim.step();
	memsim.step();
	memsim.step();
	
	return 0;
}
