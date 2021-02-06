#pragma once

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace vortex {

class ArchDef {
public:
  struct Undefined {};

  ArchDef(const std::string &s, 
          int num_cores, 
          int num_warps, 
          int num_threads) {
    std::istringstream iss(s.c_str());          
    wordSize_   = 4;
    encChar_    = 'w';
    numRegs_    = 32;
    numPRegs_   = 0;
    numCores_   = num_cores;
    numWarps_   = num_warps;
    numThreads_ = num_threads;      
    extent_     = EXT_END;
  }

  operator std::string () const {
    if (extent_ == EXT_NULL) 
      return "";

    std::ostringstream oss;
    if (extent_ >= EXT_WORDSIZE) oss << wordSize_;
    if (extent_ >= EXT_ENC     ) oss << encChar_;
    if (extent_ >= EXT_REGS    ) oss << numRegs_;
    if (extent_ >= EXT_PREGS   ) oss << '/' << numPRegs_;
    if (extent_ >= EXT_THREADS ) oss << '/' << numThreads_;
    if (extent_ >= EXT_WARPS   ) oss << '/' << numWarps_;
    if (extent_ >= EXT_CORES   ) oss << '/' << numCores_;

    return oss.str();
  }

  bool operator==(const ArchDef &r) const {
    Extent minExtent(r.extent_ > extent_ ? extent_ : r.extent_);

    // Can't be equal if we can't specify a binary encoding at all.
    if (minExtent < EXT_PREGS) 
      return false;
    
    if (minExtent >= EXT_WORDSIZE) { 
      if (wordSize_!=r.wordSize_) 
        return false; 
    }

    if (minExtent >= EXT_ENC) { 
      if (encChar_ != r.encChar_) 
        return false; 
    }
    
    if (minExtent >= EXT_REGS) { 
      if (numRegs_   !=   r.numRegs_) 
        return false; 
    }
    
    if (minExtent >= EXT_PREGS) { 
      if (numPRegs_  !=  r.numPRegs_) 
        return false; 
    }
    
    if (minExtent >= EXT_THREADS) { 
      if (numThreads_ != r.numThreads_) 
        return false; 
    }
    
    if (minExtent >= EXT_WARPS) { 
      if (numWarps_  !=  r.numWarps_) 
        return false; 
    }

    if (minExtent >= EXT_CORES) { 
      if (numCores_  !=  r.numCores_) 
        return false; 
    }

    return true;
  }

  bool operator!=(const ArchDef &r) const { 
    return !(*this == r); 
  }

  Size getWordSize() const { 
    if (extent_ <  EXT_WORDSIZE) 
      throw Undefined(); 
    return wordSize_; 
  }

  char getEncChar() const {
    if ((extent_ < EXT_ENC) || (encChar_ == 'x'))
      throw Undefined(); 
    return encChar_;
  }

  RegNum getNumRegs() const {
    if (extent_ < EXT_REGS) 
      throw Undefined(); 
    return numRegs_;
  }

  RegNum getNumPRegs() const {
    if (extent_ < EXT_PREGS) 
      throw Undefined(); 
    return numPRegs_;
  }

  ThdNum getNumThreads() const {
    if (extent_ < EXT_THREADS) 
      throw Undefined(); 
    return numThreads_;
  }

  ThdNum getNumWarps() const {
    if (extent_ < EXT_WARPS) 
      throw Undefined(); 
    return numWarps_;
  }

  ThdNum getNumCores() const {
    if (extent_ < EXT_CORES) 
      throw Undefined(); 
    return numCores_;
  }

  bool is_cpu_mode() const {
    return cpu_mode_;
  }
  
private:
  enum Extent { 
    EXT_NULL, 
    EXT_WORDSIZE, 
    EXT_ENC, 
    EXT_REGS, 
    EXT_PREGS, 
    EXT_THREADS,
    EXT_WARPS,
    EXT_CORES,
    EXT_END
  };

  Extent extent_;
  Size wordSize_;
  ThdNum numThreads_;    
  ThdNum numWarps_;
  ThdNum numCores_;
  RegNum numRegs_;
  ThdNum numPRegs_;
  char encChar_;
  bool cpu_mode_;
};

}