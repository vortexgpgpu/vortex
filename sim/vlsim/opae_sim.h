#pragma once

#include <VX_config.h>
#include <vortex_afu.h>

#include <ostream>
#include <future>
#include <list>
#include <unordered_map>

#ifndef MEMORY_BANKS 
  #ifdef PLATFORM_PARAM_LOCAL_MEMORY_BANKS
    #define MEMORY_BANKS PLATFORM_PARAM_LOCAL_MEMORY_BANKS
  #else
    #define MEMORY_BANKS 2
  #endif
#endif

#undef MEM_BLOCK_SIZE
#define MEM_BLOCK_SIZE    (PLATFORM_PARAM_LOCAL_MEMORY_DATA_WIDTH / 8)

#define CACHE_BLOCK_SIZE  64

namespace vortex {

class VL_OBJ;
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

  typedef struct {
    int cycles_left;  
    std::array<uint8_t, MEM_BLOCK_SIZE> data;
    uint32_t addr;
  } mem_rd_req_t;

  typedef struct {
    int cycles_left;  
    std::array<uint8_t, CACHE_BLOCK_SIZE> data;
    uint64_t addr;
    uint32_t mdata;
  } cci_rd_req_t;

  typedef struct {
    int cycles_left;  
    uint32_t mdata;
  } cci_wr_req_t;

  typedef struct {    
    uint64_t* data;
    size_t    size;
    uint64_t  ioaddr;  
  } host_buffer_t;

  void reset();

  void eval();

  void step();

  void sRxPort_bus();
  void sTxPort_bus();
  void avs_bus();

  std::future<void> future_;
  bool stop_;

  std::unordered_map<int64_t, host_buffer_t> host_buffers_;
  int64_t host_buffer_ids_;

  std::list<mem_rd_req_t> mem_reads_ [MEMORY_BANKS];

  std::list<cci_rd_req_t> cci_reads_;

  std::list<cci_wr_req_t> cci_writes_;

  std::mutex mutex_;

  RAM *ram_;

  VL_OBJ* vl_obj_;
};

}