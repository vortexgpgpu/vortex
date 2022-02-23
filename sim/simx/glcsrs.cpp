#include "glcsrs.h"

using namespace vortex;

GlobalCSRS::GlobalCSRS() {
  this->clear();
}

GlobalCSRS::~GlobalCSRS() {}

void GlobalCSRS::clear() {
  tex_csrs.clear();
  raster_csrs.clear();
  rop_csrs.clear();
}

void GlobalCSRS::write(uint32_t addr, uint64_t value) {     
  if (addr >= CSR_TEX_STATE_BEGIN
   && addr < CSR_TEX_STATE_END) {
      tex_csrs.write(addr, value);
      return;
  }

  if (addr >= CSR_RASTER_STATE_BEGIN
   && addr < CSR_RASTER_STATE_END) { 
    raster_csrs.write(addr, value);
    return;
  }

  if (addr >= CSR_ROP_STATE_BEGIN
   && addr < CSR_ROP_STATE_END) {
    rop_csrs.write(addr, value);
    return;
  }

  std::cout << std::hex << "Error: invalid global CSR read addr=0x" << addr << ", value=0x" << value << std::endl;
  std::abort();
}