

#include <iostream>
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