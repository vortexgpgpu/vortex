#include "VVX_generic_queue.h"
#include "VVX_generic_queue__Syms.h"
#include "verilated.h"

#include <verilated_vcd_c.h>

#include <iostream>
#include <vector>

uint64_t timestamp = 0;

double sc_time_stamp() { 
  return timestamp;
}

int main(int argc, char **argv){
	Verilated::commandArgs(argc, argv); //passes the command args to the object

	VVX_generic_queue *tb = new VVX_generic_queue;
	tb->reset = 1; 
	tb->eval(); 
	tb->reset = 0; 
	unsigned int clk;
	bool full;
	bool empty;
	int size; 
	int data_out; 
	tb->data_in = 0xff; 
	
	tb->push = 1; 

	for (int i = 0; i < 5; ++i){
		//toggle the clock
		tb->eval();
		tb->clk = 1; 

		tb->eval();
		tb->clk = 0;
		tb->eval();


		full = tb->full; 
		empty = tb->empty;
		size = tb->size;
		data_out = tb->data_out;
		clk = tb->clk; 

		std::cout << "clk: " << clk << std::endl; 
		std::cout << "data_out: " << data_out << std::endl; 
		std::cout << "empty: " << empty << std::endl; 
		std::cout << "full: " << full << std::endl; 
		std::cout << "size: " << size << std::endl; 
		
	}

	delete tb;
	
	exit(0);
}
