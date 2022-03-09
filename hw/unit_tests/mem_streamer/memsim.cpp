#include <random>
#include "memsim.h"

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

//////////////////////////////////////////////////////

int generate_rand (int min, int max) {
	int range = max - min + 1;
	return rand() % range + min;
}

//////////////////////////////////////////////////////

MemSim::MemSim() {
	msu_ = new VVX_mem_streamer();

	mem_req_ 	= new mem_req_t;
	mem_rsp_ 	= new mem_rsp_t;

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

void MemSim::attach_core() {
	if (msu_->req_ready) {
		msu_->req_valid 	= generate_rand(0, 1);
		msu_->req_rw 		= false;
		msu_->req_mask 		= generate_rand(0, 0b1111);
		msu_->req_byteen 	= 0b1;
		msu_->req_addr 		= generate_rand(0, 0x10000000);
		msu_->req_data 		= 0x00000000;
		msu_->req_tag 		= generate_rand(0, 0xFF);
	}
	msu_->rsp_ready = true;
}

void MemSim::attach_ram () {
	bool is_duplicate = false;

	// Check if ram is full/busy
	msu_->mem_req_ready = 0b1111;

	// Simulate 4 cycle delay in response
	int dequeue_index = -1;
	for (int i = 0; i < ram_.size(); i++) {
		if (ram_[i].tick > 0) {
			ram_[i].tick -= 1;	
		}
		if (dequeue_index == -1 && ram_[i].tick == 0) {
			dequeue_index = i;
		}
		std::cout<<ram_[i].tick<<"\n";
	}

	// Time to respond to the request
	if (dequeue_index != -1) {
		msu_->mem_rsp_valid = 1;
		msu_->mem_rsp_mask 	= generate_rand (0, ram_[dequeue_index].valid);
		msu_->mem_rsp_data 	= generate_rand (0x20000000, 0x30000000);
		msu_->mem_rsp_tag 	= ram_[dequeue_index].tag;
		ram_.erase(ram_.begin() + dequeue_index);
	}

	// Store incoming request
	if (msu_->mem_req_valid && !is_duplicate) {
		mem_req_t mem_req;
		mem_req.valid 	= msu_->mem_req_valid;
		mem_req.rw 		= msu_->mem_req_rw;
		mem_req.byteen	= msu_->mem_req_byteen;
		mem_req.addr 	= msu_->mem_req_addr;
		mem_req.data 	= msu_->mem_req_data;
		mem_req.tag 	= msu_->mem_req_tag;
		mem_req.tick 	= MEM_LATENCY;
		ram_.push_back(mem_req);
	}
}

void MemSim::run() {
	this->reset();

	while (sc_time_stamp() < SIM_TIME) {
		this->step();
		this->attach_core();
		this->attach_ram();
	}
}

//////////////////////////////////////////////////////

int main (int argc, char** argv, char** env) {
    Verilated::commandArgs(argc, argv);

	MemSim memsim;
	memsim.run();

	return 0;
}

//////////////////////////////////////////////////////
