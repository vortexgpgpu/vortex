
#pragma once

#include <memory>
#include <util.h>
#include "types.h"
#include "debug.h"

namespace vortex {

class Instr;

class Pipeline {
public:
  Pipeline(const char* name);

  void clear();

  bool enter(Pipeline* drain);

  void next(Pipeline* drain);

  //--
  bool      valid;

  //--
  bool      stalled;
  bool      stall_warp;

  //--    
  int       wid;
  Word      PC;

  //--
  int       rdest_type;
  int       rdest;
  RegMask   used_iregs;
  RegMask   used_fregs;
  RegMask   used_vregs;

private:

  const char* name_;

  friend std::ostream &operator<<(std::ostream &, const Pipeline&);
};  

}