#pragma once

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace vortex {

class ArchDef {  
private:
  uint16_t num_cores_;
  uint16_t num_warps_;
  uint16_t num_threads_;
  uint16_t wsize_;
  uint16_t vsize_;
  uint16_t num_regs_;
  uint16_t num_csrs_;
  uint16_t num_barriers_;
  
public:
  ArchDef(const std::string& /*arch*/,
          uint16_t num_cores, 
          uint16_t num_warps, 
          uint16_t num_threads)   
    : num_cores_(num_cores)
    , num_warps_(num_warps)
    , num_threads_(num_threads)
    , wsize_(4)
    , vsize_(16)
    , num_regs_(32)
    , num_csrs_(4096)
    , num_barriers_(NUM_BARRIERS)
  {}

  uint16_t wsize() const { 
    return wsize_; 
  }

  uint16_t vsize() const { 
    return vsize_; 
  }

  uint16_t num_regs() const {
    return num_regs_;
  }

  uint16_t num_csrs() const {
    return num_csrs_;
  }

  uint16_t num_barriers() const {
    return num_barriers_;
  }

  uint16_t num_threads() const {
    return num_threads_;
  }

  uint16_t num_warps() const {
    return num_warps_;
  }

  uint16_t num_cores() const {
    return num_cores_;
  }
};

}