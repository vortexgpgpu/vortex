#include "dcrs.h"

using namespace vortex;

void DCRS::write(uint32_t addr, uint32_t value) {     
  if (addr >= VX_DCR_BASE_STATE_BEGIN
   && addr < VX_DCR_BASE_STATE_END) {
      base_dcrs.write(addr, value);
      return;
  }

  if (addr >= VX_DCR_TEX_STATE_BEGIN
   && addr < VX_DCR_TEX_STATE_END) {
      tex_dcrs.write(addr, value);
      return;
  }

  if (addr >= VX_DCR_RASTER_STATE_BEGIN
   && addr < VX_DCR_RASTER_STATE_END) { 
    raster_dcrs.write(addr, value);
    return;
  }

  if (addr >= VX_DCR_ROP_STATE_BEGIN
   && addr < VX_DCR_ROP_STATE_END) {
    rop_dcrs.write(addr, value);
    return;
  }

  std::cout << std::hex << "Error: invalid global DCR addr=0x" << addr << std::endl;
  std::abort();
}