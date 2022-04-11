#include "dcrs.h"

using namespace vortex;

DCRS::DCRS() {
  this->clear();
}

DCRS::~DCRS() {}

void DCRS::clear() {
  base_dcrs.clear();
  tex_dcrs.clear();
  raster_dcrs.clear();
  rop_dcrs.clear();
}

void DCRS::write(uint32_t addr, uint64_t value) {     
  if (addr >= DCR_BASE_STATE_BEGIN
   && addr < DCR_BASE_STATE_END) {
      base_dcrs.write(addr, value);
      return;
  }

  if (addr >= DCR_TEX_STATE_BEGIN
   && addr < DCR_TEX_STATE_END) {
      tex_dcrs.write(addr, value);
      return;
  }

  if (addr >= DCR_RASTER_STATE_BEGIN
   && addr < DCR_RASTER_STATE_END) { 
    raster_dcrs.write(addr, value);
    return;
  }

  if (addr >= DCR_ROP_STATE_BEGIN
   && addr < DCR_ROP_STATE_END) {
    rop_dcrs.write(addr, value);
    return;
  }

  std::cout << std::hex << "Error: invalid global DCR read addr=0x" << addr << ", value=0x" << value << std::endl;
  std::abort();
}