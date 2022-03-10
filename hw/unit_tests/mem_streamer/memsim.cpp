#include <random>
#include "memsim.h"
#include "ram.h"

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

	mem_rsp_active_ = false;
	mem_rsp_stall_ = false;

	// Enable tracing
	Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC;
    msu_->trace(trace_, 99);
    trace_->open("trace.vcd");
}

//////////////////////////////////////////////////////

MemSim::~MemSim() {
	trace_->close();
	delete msu_;
}

//////////////////////////////////////////////////////

void MemSim::eval() {
	msu_->eval();
	trace_->dump(timestamp++);
}

//////////////////////////////////////////////////////

void MemSim::step() {
	msu_->clk = 0;
	this->eval();

	msu_->clk = 1;
	this->eval();
}

//////////////////////////////////////////////////////

void MemSim::reset() {
	msu_->reset = 1;
	this->step();

	msu_->reset = 0;
	this->step();

}

//////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////

void MemSim::attach_ram2 (RAM *ram) {

	req_t req;
	req.valid 			= msu_->mem_req_valid;
	req.rw 				= msu_->mem_req_rw;
	req.byteen			= msu_->mem_req_byteen;
	req.addr 			= msu_->mem_req_addr;
	req.data 			= msu_->mem_req_data;
	req.tag 			= msu_->mem_req_tag;

	msu_->mem_req_ready = ram->insert_req(req);
	std::cout<<"SIM: Request ready: "<<+msu_->mem_req_ready<<std::endl;

	rsp_t rsp;
	rsp = ram->schedule_rsp();

	msu_->mem_rsp_valid = rsp.valid;
	msu_->mem_rsp_mask 	= rsp.mask;
	msu_->mem_rsp_data 	= rsp.data;
	msu_->mem_rsp_tag 	= rsp.tag;
	rsp.ready 			= msu_->mem_rsp_ready;

	ram->halt_rsp(rsp);
}

//////////////////////////////////////////////////////

void MemSim::run(RAM *ram) {
	this->reset();

	while (sc_time_stamp() < SIM_TIME) {
		std::cout<<"========================="<<"\n";
		std::cout<<"Cycle: "<<sc_time_stamp()<<"\n";
		this->step();
		this->attach_core();
		this->attach_ram2(ram);
	}
}

//////////////////////////////////////////////////////

int main (int argc, char** argv, char** env) {
    Verilated::commandArgs(argc, argv);

	MemSim memsim;
	RAM ram;	

	memsim.run(&ram);

	return 0;
}

//////////////////////////////////////////////////////



/* Code dump

void MemSim::attach_ram () {
	bool is_duplicate = false;

	msu_->mem_req_ready = generate_rand(10, 15);
	int dequeue_index = -1;

	std::cout<<"Num entries in RAM: "<<ram_.size()<<"\n";

	// Simulate 4 cycle delay in response
	for (int i = 0; i < ram_.size(); i++) {
		if (!mem_rsp_stall_) {
			if (ram_[i].cycles_left > 0) {
				ram_[i].cycles_left -= 1;	
			}
		}
		
		if (dequeue_index == -1 && ram_[i].cycles_left == 0) {
			dequeue_index = i;
		}
		std::cout<<"Cycles left: "<<ram_[i].cycles_left<<"\n";
	}

	for (int i = 0; i < ram_.size(); i++) {
		if (msu_->mem_req_addr == ram_[i].addr) {
			is_duplicate = true;
			break;
		}
	}

	if (mem_rsp_active_ && msu_->mem_rsp_valid && msu_->mem_rsp_ready) {
    	mem_rsp_active_ = false;
  	}

	// Time to respond to the request
	if (!mem_rsp_active_) {
		if (dequeue_index != -1) {
			std::cout<<"Scheduling response\n";
			msu_->mem_rsp_valid = 1;
			msu_->mem_rsp_mask 	= generate_rand(ram_[dequeue_index].rsp_sent_mask, ram_[dequeue_index].valid);
			msu_->mem_rsp_data 	= generate_rand(0x20000000, 0x30000000);
			msu_->mem_rsp_tag 	= ram_[dequeue_index].tag;

			std::cout<<std::hex;
			std::cout<<"Valid: "<<+msu_->mem_rsp_valid<<"\n";
			std::cout<<"Response mask: "<<+msu_->mem_rsp_mask<<" Required mask: "<<+ram_[dequeue_index].valid<<"\n";

			if (msu_->mem_rsp_mask == ram_[dequeue_index].valid) {
				std::cout<<"Erasing entry after all requests have been processed\n";
				ram_.erase(ram_.begin() + dequeue_index);
				mem_rsp_stall_ = false;
			} else {
				mem_rsp_stall_ = true;
				ram_[dequeue_index].rsp_sent_mask = msu_->mem_rsp_mask;
				std::cout<<"Stall\n";
			}
			mem_rsp_active_ = true;

		} else {
			msu_->mem_rsp_valid = 0;
		}
	}
	
	// Store incoming request
	if (msu_->mem_req_valid && !is_duplicate) {
		std::cout<<"Inserting new entry into RAM\n";
		mem_req_t mem_req;
		mem_req.valid 	= msu_->mem_req_valid;
		mem_req.rw 		= msu_->mem_req_rw;
		mem_req.byteen	= msu_->mem_req_byteen;
		mem_req.addr 	= msu_->mem_req_addr;
		mem_req.data 	= msu_->mem_req_data;
		mem_req.tag 	= msu_->mem_req_tag;
		mem_req.cycles_left = MEM_LATENCY;
		mem_req.rsp_sent_mask = 0;
		ram_.push_back(mem_req);
	}
}

*/