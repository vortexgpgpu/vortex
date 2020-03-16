#include <vx_driver.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "utils.h"

#define CACHE_LINESIZE		64

const char* program_file = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-f: program] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "f:h?")) != -1) {
    switch (c) {
    case 'f': {
      program_file = optarg;
    } break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  if (nullptr == program_file) {
    show_usage();
    exit(-1);
  }
}

static int upload_program(vx_device_h device, const char* filename, uint32_t transfer_size = 16 * VX_CACHE_LINESIZE) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  // allocate device buffer
  auto buffer = vx_buf_alloc(device, transfer_size);
  if (nullptr == buffer)
    return -1; 

  // get buffer address
  auto buf_ptr = (uint8_t*)vs_buf_ptr(buffer);

  //
  // copy initialization routine
  //

  ((uint32_t*)buf_ptr)[0] = 0xf1401073;
  ((uint32_t*)buf_ptr)[1] = 0xf1401073;      
  ((uint32_t*)buf_ptr)[2] = 0x30101073;
  ((uint32_t*)buf_ptr)[3] = 0x800000b7;
  ((uint32_t*)buf_ptr)[4] = 0x000080e7;

  vx_copy_to_fpga(buffer, 0, 5 * 4, 0);
  
  //
  // copy hex program
  //

  char line[ihex_t::MAX_LINE_SIZE];
  uint32_t hex_offset = 0;
  uint32_t prev_hex_address = 0;
  uint32_t dest_address = -1;
  uint32_t src_offset = 0;

  while (true) {
    ifs.getline(line, ihex_t::MAX_LINE_SIZE);
    if (!ifs) 
      break;

    ihex_t ihex;
    parse_ihex_line(line, &ihex);
    if (ihex.is_eof)
      break;

    if (ihex.has_offset) {
      hex_offset = ihex.offset;
    }

    if (ihex.data_size != 0) {
        auto hex_address = ihex.address + hex_offset;
        if (dest_address == (uint32_t)-1) {
          dest_address = (hex_address / VX_CACHE_LINESIZE) * VX_CACHE_LINESIZE;          
          src_offset = hex_address - dest_address;
        } else {
          auto delta = hex_address - prev_hex_address;
          src_offset += delta;
        }
        for (uint32_t i = 0; i < ihex.data_size; ++i) {          
          if (src_offset >= transfer_size) {
            // flush current batch to FPGA
            vx_copy_to_fpga(buffer, dest_address, transfer_size, 0);
            dest_address = (hex_address/ VX_CACHE_LINESIZE) * VX_CACHE_LINESIZE;
            src_offset = hex_address - dest_address;            
          }
          buf_ptr[src_offset++] = ihex.data[i];
          ++hex_address;
        }
        prev_hex_address = hex_address;
    }
  }

  // flush last batch to FPGA
  if (src_offset) {
    vx_copy_to_fpga(buffer, dest_address, src_offset, 0);
  }

  vx_buf_release(buffer);

  return 0;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  auto device = vx_dev_open();

  // upload program
  if (0 != upload_program(device, program_file)) {
    vx_dev_close(device);
    return -1;  
  }

  // start device
  if (0 != vx_start(device)) {
    vx_dev_close(device);
    return -1;  
  }

  // wait for completion
  if (0 != vx_ready_wait(device, -1)) {
    vx_dev_close(device);
    return -1;  
  }
  
  // close device
  vx_dev_close(device);

  return 0;
}