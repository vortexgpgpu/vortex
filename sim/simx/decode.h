#pragma once

#include <vector>
#include <memory>

namespace vortex {

class Arch;
class Instr;

class Decoder {
public:
  Decoder(const Arch &);    
  
  std::shared_ptr<Instr> decode(uint32_t code) const;
};

}