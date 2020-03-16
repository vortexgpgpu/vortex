#include <iostream>
#include <fstream>
#include "utils.h"

static uint32_t hti_old(char c) {
      if (c >= 'A' && c <= 'F')
          return c - 'A' + 10;
      if (c >= 'a' && c <= 'f')
          return c - 'a' + 10;
      return c - '0';
  }

static uint32_t hToI_old(char *c, uint32_t size) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < size; i++) {
        value += hti_old(c[i]) << ((size - i - 1) * 4);
    }
    return value;
}


int parse_ihex_line(char* line, ihex_t* out) {
    if (line[0] != ':') {
      std::cout << "error: invalid line entry!" << std::endl;  
      return -1;
    }
    
    uint32_t data_size = 0;
    uint32_t address = 0;    
    uint32_t offset = 0;
    bool has_offset = false;
    bool is_eof = false;

    auto record_type = hToI_old(line + 7, 2);

    switch (record_type) {
    case 0: { // data  
      data_size = hToI_old(line + 1, 2);
      address = hToI_old(line + 3, 4);
      for (uint32_t i = 0; i < data_size; i++) {
          out->data[i] = hToI_old(line + 9 + i * 2, 2); 
      }
    } break;
    case 1: // end of file
      is_eof = true;
      break;
    case 2: // extended segment address
      offset = hToI_old(line + 9, 4) << 4;
      has_offset = true;
      break;
    case 3: // start segment address
      break;
    case 4: // extended linear address
      offset = hToI_old(line + 9, 4) << 16;
      has_offset = true;
      break;
    case 5: // start linear address
      break;
    default:
      return -1;
    }

    out->address = address;
    out->data_size = data_size;
    out->offset = offset;
    out->has_offset = has_offset;
    out->is_eof = is_eof;

    return 0;
  }

  int upload_program(vx_device_h device, const char* filename) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  uint32_t transfer_size = 16 * VX_CACHE_LINESIZE;

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