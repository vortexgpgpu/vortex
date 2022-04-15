#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "rop_unit.h"
#include "pipeline.h"

namespace vortex {

class Core;
class RopUnit;

class RopSvc : public SimObject<RopSvc> {
public:
  
  SimPort<pipeline_trace_t*> Input;
  SimPort<pipeline_trace_t*> Output;

  RopSvc(const SimContext& ctx, 
          const char* name,  
          Core* core,
          RopUnit::Ptr rop_unit);    

  ~RopSvc();

  void reset();

  uint32_t csr_read(uint32_t wid, uint32_t tid, uint32_t addr);
  
  void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

  void write(uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data);

  void tick();
  
private:

  class Impl;
  Impl* impl_;
};

}