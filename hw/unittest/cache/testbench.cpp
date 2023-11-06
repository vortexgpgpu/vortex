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
#include <iostream>
#include <fstream>
#include <iomanip>

#define VCD_OUTPUT 1


int REQ_RSP(CacheSim *sim){ //verified
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

  int check = sim->assert_equal(data, write->tag);
  
  if (check == 4) return 1; 

  return 0;
}

int HIT_1(CacheSim *sim){
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
  write->tag = 0x11;

  //read req
  core_req_t* read = new core_req_t;
  read->valid = 0xf;
  read->rw = 0;
  read->byteen = 0xffff;
  read->addr = addr;
  read->data = addr; 
  read->tag = 0x22;
  
  // reset the device
  sim->reset();

  //queue reqs
  sim->send_req(write);
  sim->send_req(read);

  sim->run();

  bool check = sim->assert_equal(data, write->tag);
  
  return check;
}

int MISS_1(CacheSim *sim){
  unsigned int addr1[4] = {0x12222222, 0xabbbbbbb, 0xcddddddd, 0xe4444444};
  unsigned int addr2[4] = {0x12229222, 0xabbbb4bb, 0xcddd47dd, 0xe4423544};
  unsigned int addr3[4] = {0x12223332, 0xabb454bb, 0xcdddeefd, 0xe4447744};
  unsigned int data[4] = {0xffffffff, 0x11111111, 0x22222222, 0x33333333};
  unsigned int rsp[4] = {0,0,0,0};
  char responded = 0;
  //write req
  core_req_t* write = new core_req_t;
  write->valid = 0xf;
  write->rw = 0xf;
  write->byteen = 0xffff;
  write->addr = addr1; 
  write->data = data;
  write->tag = 0xff;

  //read req
  core_req_t* read1 = new core_req_t;
  read1->valid = 0xf;
  read1->rw = 0;
  read1->byteen = 0xffff;
  read1->addr = addr1;
  read1->data = data; 
  read1->tag = 0xff;

  core_req_t* read2 = new core_req_t;
  read2->valid = 0xf;
  read2->rw = 0;
  read2->byteen = 0xffff;
  read2->addr = addr2;
  read2->data = data; 
  read2->tag = 0xff;
  
  core_req_t* read3 = new core_req_t;
  read3->valid = 0xf;
  read3->rw = 0;
  read3->byteen = 0xffff;
  read3->addr = addr3;
  read3->data = data; 
  read3->tag = 0xff;
  
  // reset the device
  sim->reset();

  //queue reqs
  sim->send_req(write);
  sim->send_req(read1);
  sim->send_req(read2);
  sim->send_req(read3);

  sim->run();

  bool check = sim->assert_equal(data, write->tag);
  
  return check;
}
int FLUSH(CacheSim *sim){
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
  //happens whenever the core is stalled or memory is stalled
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
