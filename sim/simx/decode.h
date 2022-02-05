#pragma once

#include <vector>
#include <memory>

namespace vortex {

class ArchDef;
class Instr;

class Decoder {
public:
  Decoder(const ArchDef &);    
  
  std::shared_ptr<Instr> decode(uint32_t code) const;
};

}