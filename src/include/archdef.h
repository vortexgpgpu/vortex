/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __ARCHDEF_H
#define __ARCHDEF_H

#include <iostream>
#include <string>
#include <sstream>

#include "types.h"

namespace Harp {
  class ArchDef {
  public:
    struct Undefined {};

    ArchDef(const std::string &s) {
      std::cout << "New archdef for \"" << s << "\"\n";
      std::istringstream iss(s.c_str());
      
      iss >> wordSize;
      if (!iss) { extent = EXT_NULL; return; }
      iss >> encChar;
      if (!iss) { extent = EXT_WORDSIZE; return; }
      iss >> nRegs;
      if (!iss) { extent = EXT_ENC; return; }
      char sep;
      iss >> sep >> nPRegs;
      if (!iss || sep != '/') { extent = EXT_REGS; return; }
      iss >> sep >> nThds;
      if (!iss || sep != '/') { extent = EXT_PREGS; return; }
      extent = EXT_THDS;

      std::cout << nRegs << " regs, " << nPRegs << " pred regs.\n";
    }

    operator std::string () const {
      if (extent == EXT_NULL) return "";

      std::ostringstream oss;
      if (extent >= EXT_WORDSIZE) oss << wordSize;
      if (extent >= EXT_ENC     ) oss << encChar;
      if (extent >= EXT_REGS    ) oss << nRegs;
      if (extent >= EXT_PREGS   ) oss << '/' << nPRegs;
      if (extent >= EXT_THDS    ) oss << '/' << nThds;

      return oss.str();
    }

    bool operator==(const ArchDef &r) const {
      return (extent == r.extent)   && (wordSize == r.wordSize) && 
             (encChar == r.encChar) && (nRegs == r.nRegs) &&
             (nPRegs == r.nPRegs)   && (nThds == r.nThds);
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
    
  private:
    enum { 
      EXT_NULL, EXT_WORDSIZE, EXT_ENC, EXT_REGS, EXT_PREGS, EXT_THDS
    } extent;

    Size wordSize;
    ThdNum nThds;
    RegNum nRegs, nPRegs;
    char encChar;
  };
};

#endif
