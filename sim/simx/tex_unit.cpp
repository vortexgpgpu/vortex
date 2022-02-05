#include "tex_unit.h"
#include "core.h"
#include <texturing.h>
#include <VX_config.h>

using namespace vortex;
using namespace cocogfx;

enum class FilterMode {
  Point,
  Bilinear,
  Trilinear,
};

TexUnit::TexUnit(Core* core) : core_(core) {}

TexUnit::~TexUnit() {}

void TexUnit::clear() {
  for (auto& state : states_) {
    state = 0;
  }
}

uint32_t TexUnit::get_state(uint32_t state) {
  return states_.at(state);
}
  
void TexUnit::set_state(uint32_t state, uint32_t value) {
  states_.at(state) = value;
}

uint32_t TexUnit::read(int32_t u, 
                       int32_t v, 
                       int32_t lod, 
                       std::vector<mem_addr_size_t>* mem_addrs) {
  //--
  auto xu = Fixed<TEX_FXD_FRAC>::make(u);
  auto xv = Fixed<TEX_FXD_FRAC>::make(v);
  uint32_t base_addr  = states_.at(TEX_STATE_ADDR) + states_.at(TEX_STATE_MIPOFF(lod));
  uint32_t log_width  = std::max<int32_t>(states_.at(TEX_STATE_WIDTH) - lod, 0);
  uint32_t log_height = std::max<int32_t>(states_.at(TEX_STATE_HEIGHT) - lod, 0);
  auto format         = (TexFormat)states_.at(TEX_STATE_FORMAT);    
  auto filter         = (FilterMode)states_.at(TEX_STATE_FILTER);    
  auto wrapu          = (WrapMode)states_.at(TEX_STATE_WRAPU);
  auto wrapv          = (WrapMode)states_.at(TEX_STATE_WRAPV);

  auto stride = Stride(format);
  
  switch (filter) {
  case FilterMode::Bilinear: {
    // addressing
    uint32_t offset00, offset01, offset10, offset11;
    uint32_t alpha, beta;
    TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv, 
      &offset00, &offset01, &offset10, &offset11, &alpha, &beta);

    uint32_t addr00 = base_addr + offset00 * stride;
    uint32_t addr01 = base_addr + offset01 * stride;
    uint32_t addr10 = base_addr + offset10 * stride;
    uint32_t addr11 = base_addr + offset11 * stride;

    // memory lookup
    uint32_t texel00(0), texel01(0), texel10(0), texel11(0);
    core_->dcache_read(&texel00, addr00, stride);
    core_->dcache_read(&texel01, addr01, stride);
    core_->dcache_read(&texel10, addr10, stride);
    core_->dcache_read(&texel11, addr11, stride);

    mem_addrs->push_back({addr00, stride});
    mem_addrs->push_back({addr01, stride});
    mem_addrs->push_back({addr10, stride});
    mem_addrs->push_back({addr11, stride});

    // filtering
    auto color = TexFilterLinear(
      format, texel00, texel01, texel10, texel11, alpha, beta);
    return color;
  }
  case FilterMode::Point: {
    // addressing
    uint32_t offset;
    TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);
    
    uint32_t addr = base_addr + offset * stride;

    // memory lookup
    uint32_t texel(0);
    core_->dcache_read(&texel, addr, stride);
    mem_addrs->push_back({addr, stride});

    // filtering
    auto color = TexFilterPoint(format, texel);
    return color;
  }
  default:
    std::abort();
    return 0;
  }
}