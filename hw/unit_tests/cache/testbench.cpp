#include "cachesim.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define VCD_OUTPUT 1

int main(int argc, char **argv)
{
  //init
  RAM ram;
	CacheSim cachesim;
  cachesim.attach_ram(&ram);

  unsigned int addr[4] = {0x12222222, 0xabbbbbbb, 0xcddddddd, 0xe4444444};
  unsigned int data[4] = {0xffffffff, 0x11111111, 0x22222222, 0x33333333};
  //write req
  core_req_t* write = new core_req_t;
  write->valid = 0xf;
  write->rw = 0xf;
  write->byteen = 0xffff;
  write->addr = addr; 
  write->data = data;
  write->tag = 0xff;

  //read req
  core_req_t* read = new core_req_t;
  read->valid = 0xf;
  read->rw = 0;
  read->byteen = 0xffff;
  read->addr = addr;
  read->data = addr; 
  read->tag = 0xff;
  
	// reset the device
  cachesim.reset();

  //queue reqs
  cachesim.send_req(write);
  cachesim.send_req(read);
  cachesim.step();
  //cachesim.get_core_req();
  //write block to cache
 // cachesim.set_core_req();

  for (int i = 0; i < 100; ++i){
    /*if(i == 1){
      cachesim.clear_req();
    }*/
    cachesim.step();
  }
  cachesim.get_core_req();
  // read block
  //cachesim.set_core_req2();
  for (int i = 0; i < 100; ++i){
    if(i == 1){
      //read block from cache
      cachesim.clear_req();

    }
    cachesim.step();
  } 
  
	return 0;
}
