#pragma once

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace vortex {

class ArchDef {
public:
  ArchDef(const std::string &/*arch*/,
          int num_cores, 
          int num_warps, 
          int num_threads) {         
    wsize_       = 4;
    vsize_       = 16;
    num_regs_    = 32;
    num_csrs_    = 4096;
    num_barriers_= NUM_BARRIERS;
    num_cores_   = num_cores;
    num_warps_   = num_warps;
    num_threads_ = num_threads;
  }

  int wsize() const { 
    return wsize_; 
  }

  int vsize() const { 
    return vsize_; 
  }

  int num_regs() const {
    return num_regs_;
  }

  int num_csrs() const {
    return num_csrs_;
  }

  int num_barriers() const {
    return num_barriers_;
  }

  int num_threads() const {
    return num_threads_;
  }

  int num_warps() const {
    return num_warps_;
  }

  int num_cores() const {
    return num_cores_;
  }
  
private:

  int wsize_;
  int vsize_;
  int num_regs_;
  int num_csrs_;
  int num_barriers_;
  int num_threads_;
  int num_warps_;
  int num_cores_;
};

}