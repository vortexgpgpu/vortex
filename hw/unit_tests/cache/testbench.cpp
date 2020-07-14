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
  
  
	// reset the device
  cachesim.reset();

  //write block to cache
  cachesim.set_core_req();

  for (int i = 0; i < 100; ++i){
    /*if(i == 1){
      cachesim.clear_req();
    }*/
    cachesim.step();
    cachesim.get_core_rsp();
  }
  
  // read block
  cachesim.set_core_req2();
  for (int i = 0; i < 100; ++i){
    /*if(i == 1){
      //read block from cache
      cachesim.clear_req();

    }*/
    cachesim.step();
    cachesim.get_core_rsp();
  } 
  
  /*
  core_req_t *write;
  write->valid = 1;
  //write.tag = 0xff; //TODO: make a reasonable tag
  //write.addr[0] = 0x11111111; 
  //write.addr[1] = 0x22222222;
  //write.addr[2] = 0x33333333;
  //write.addr[3] = 0x44444444;
  //write.
  */
	return 0;
}
