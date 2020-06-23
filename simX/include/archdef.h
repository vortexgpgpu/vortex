/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __ARCHDEF_H
#define __ARCHDEF_H

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace Harp {
  class ArchDef {
  public:
    struct Undefined {};

<<<<<<< HEAD
    ArchDef(const std::string &s, bool cpu_mode = false) 
      : cpu_mode_(cpu_mode) {
=======
    ArchDef(const std::string &s, int num_warps = 32, int num_threads = 32) {
>>>>>>> fpga_synthesis
      std::istringstream iss(s.c_str());
            
      wordSize = 4;
      encChar = 'w';
      nRegs = 32;
      nPRegs = 0;
      nWarps = num_warps;
      nThds = num_threads;
      
      extent = EXT_WARPS;

      // if (!iss) { extent = EXT_NULL; return; }
      // iss >> encChar;
      // if (!iss) { extent = EXT_WORDSIZE; return; }
      // iss >> nRegs;
      // if (!iss) { extent = EXT_ENC; return; }
      // char sep;
      // iss >> sep >> nPRegs;
      // if (!iss || sep != '/') { extent = EXT_REGS; return; }
      // iss >> sep >> nThds;
      // if (!iss || sep != '/') { extent = EXT_PREGS; return; }
      // iss >> sep >> nWarps;
      // if (!iss || sep != '/') { extent = EXT_THDS; return; }
      // extent = EXT_WARPS;
    }

    operator std::string () const {
      if (extent == EXT_NULL) return "";

      std::ostringstream oss;
      if (extent >= EXT_WORDSIZE) oss << wordSize;
      if (extent >= EXT_ENC     ) oss << encChar;
      if (extent >= EXT_REGS    ) oss << nRegs;
      if (extent >= EXT_PREGS   ) oss << '/' << nPRegs;
      if (extent >= EXT_THDS    ) oss << '/' << nThds;
      if (extent >= EXT_WARPS    ) oss << '/' << nWarps;

      return oss.str();
    }

    bool operator==(const ArchDef &r) const {
      Extent minExtent(r.extent > extent ? extent : r.extent);

      // Can't be equal if we can't specify a binary encoding at all.
      if (minExtent < EXT_PREGS) return false;
      
      if (minExtent >= EXT_WORDSIZE) { if (wordSize!=r.wordSize) return false; }
      if (minExtent >= EXT_ENC     ) { if (encChar != r.encChar) return false; }
      if (minExtent >= EXT_REGS    ) { if (nRegs   !=   r.nRegs) return false; }
      if (minExtent >= EXT_PREGS   ) { if (nPRegs  !=  r.nPRegs) return false; }
      if (minExtent >= EXT_THDS    ) { if (nThds   !=   r.nThds) return false; }
      if (minExtent >= EXT_WARPS   ) { if (nWarps  !=  r.nWarps) return false; }

      return true;
    }

    bool operator!=(const ArchDef &r) const { return !(*this == r); }

    Size getWordSize() const { 
      if (extent <  EXT_WORDSIZE) throw Undefined(); else return wordSize; 
    }

    char getEncChar() const {
      if (extent<EXT_ENC||encChar=='x') throw Undefined(); else return encChar;
    }

    RegNum getNRegs() const {
      if (extent < EXT_REGS) throw Undefined(); else return nRegs;
    }

    RegNum getNPRegs() const {
      if (extent < EXT_PREGS) throw Undefined(); else return nPRegs;
    }
  
    ThdNum getNThds() const {
      if (extent < EXT_THDS) throw Undefined(); else return nThds;
    }

    ThdNum getNWarps() const {
      if (extent < EXT_WARPS) throw Undefined(); else return nWarps;
    }

    bool is_cpu_mode() const {
      return cpu_mode_;
    }
    
  private:
    enum Extent { 
      EXT_NULL, EXT_WORDSIZE, EXT_ENC, EXT_REGS, EXT_PREGS, EXT_THDS, EXT_WARPS
    };

    Extent extent;

    Size wordSize;
    ThdNum nThds, nWarps;
    RegNum nRegs, nPRegs;
    char encChar;
    bool cpu_mode_;
  };
}

#endif
