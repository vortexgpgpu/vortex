#include "VVX_cache.h"
#include "VVX_cache__Syms.h"
#include "verilated.h"

#include <verilated_vcd_c.h>

#include <iostream>
#include <vector>

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

void tick(VVX_cache* tb){
	tb->eval();
	tb->clk = 1; 

	tb->eval();
	tb->clk = 0;
	tb->eval();
}

int main(int argc, char **argv){
	Verilated::commandArgs(argc, argv); //passes the command args to the object

	VVX_cache *tb = new VVX_cachee;
	
	//reset the cache
	tb->reset = 1; 
	tb->eval(); 
	tb->reset = 0; 

	//declare variables for output - data members in cache class
	unsigned int clk;
	bool full;
	bool empty;
	int size; 
	int data_out; 

	//assign inputs
	tb->core_req_valid = 1;
	tb->core_req_rw = 1; 

	char byte_en[] = {}; //word size 4 bytes
	tb->core_req_byteen[0] = 1;
	tb->core_req_byteen[1] = 1;
	tb->core_req_byteen[2] = 1;
	tb->core_req_byteen[3] = 1;

	char addr[4] = {0x1a, 0x2b, 0x3c, 0x4d}; //word addr width

	tb->core_req_addr[0] = arr[0];
	tb->core_req_addr[0] = arr[1];
	tb->core_req_addr[0] = arr[2];
	tb->core_req_addr[0] = arr[3];

	//char req_data[] = {}; //word width
	//tb->core_req_data

	//char req_tag[] = {}; //core_req_tag_count by core req_tag_width
	//tb->core_req_tag = 


	for (int i = 0; i < 5; ++i){
		//toggle the clock
		tick(tb);
	}

	delete tb;
	
	exit(0);
}

