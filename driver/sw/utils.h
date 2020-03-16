
#pragma once  

#include <vx_driver.h>

struct ihex_t {
  static constexpr int MAX_LINE_SIZE = 524;
  static constexpr int MAX_DATA_SIZE = 255;
  uint8_t data[MAX_DATA_SIZE];
  uint32_t address;    
  uint32_t data_size;    
  uint32_t offset;
  bool has_offset;
  bool is_eof;
};

int parse_ihex_line(char* line, ihex_t* out);

int upload_program(vx_device_h device, const char* filename);