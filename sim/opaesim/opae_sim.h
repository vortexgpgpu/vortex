#pragma once

#include <stdint.h>
namespace vortex {

class RAM;

class opae_sim {
public:
  
  opae_sim();
  virtual ~opae_sim();

  int prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags);

  void release_buffer(uint64_t wsid);

  void get_io_address(uint64_t wsid, uint64_t *ioaddr);

  void write_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t value);

  void read_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t *value);

private: 

  class Impl;
  Impl* impl_;  
};

}