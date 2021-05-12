
#pragma once

#include <memory>
#include "debug.h"
#include "util.h"

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

  //--
  Word      fetched;
  std::shared_ptr<Instr> instr;

private:

  const char* name_;

  friend std::ostream &operator<<(std::ostream &, const Pipeline&);
};  

}