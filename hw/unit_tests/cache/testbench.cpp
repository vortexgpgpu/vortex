#include "cachesim.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define VCD_OUTPUT 1


int REQ_RSP(CacheSim *sim){
  unsigned int addr[4] = {0x12222222, 0xabbbbbbb, 0xcddddddd, 0xe4444444};
  unsigned int data[4] = {0xffffffff, 0x11111111, 0x22222222, 0x33333333};
  unsigned int rsp[4] = {0,0,0,0};
  char responded = 0;
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
  sim->reset();

  //queue reqs
  sim->send_req(write);
  sim->send_req(read);

  sim->run();

  bool check = sim->assert_equal(data, write->tag);
  
  return check;
}

int BACK_PRESSURE(CacheSim *sim){
  unsigned int addr[4] = {0x12222222, 0xabbbbbbb, 0xcddddddd, 0xe4444444};
  unsigned int data[4] = {0xffffffff, 0x11111111, 0x22222222, 0x33333333};
  unsigned int rsp[4] = {0,0,0,0};
  char responded = 0;
  
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
  sim->reset();

  //queue reqs
  for (int i = 0; i < 10; i++){
    sim->send_req(write);
  }
  sim->send_req(read); 

  sim->run();

  bool check = sim->assert_equal(data, write->tag);
  
  return check;
}

int main(int argc, char **argv)
{
  //init
  RAM ram;
	CacheSim cachesim;
  cachesim.attach_ram(&ram);
  int check = REQ_RSP(&cachesim);
  if(check){
    std::cout << "PASSED" << std::endl;
  } else {
    std::cout << "FAILED" << std::endl;
  }
  


	return 0;
}
