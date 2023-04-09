#pragma once

#include <simobject.h>
#include "pipeline.h"
#include "cache_sim.h"
#include "tex_unit.h"
#include "raster_unit.h"
#include "rop_unit.h"

namespace vortex {

class Core;

class ExeUnit : public SimObject<ExeUnit> {
public:
    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    ExeUnit(const SimContext& ctx, Core* core, const char* name) 
        : SimObject<ExeUnit>(ctx, name) 
        , Input(this)
        , Output(this)
        , core_(core)
    {}
    
    virtual ~ExeUnit() {}

    virtual void reset() {}

    virtual void tick() = 0;

protected:
    Core* core_;
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public ExeUnit {
public:
    LsuUnit(const SimContext& ctx, Core*);

    void reset();

    void tick();

private:    
    struct pending_req_t {
      pipeline_trace_t* trace;
      uint32_t count;
    };
    HashTable<pending_req_t> pending_rd_reqs_;    
    uint32_t num_threads_;
    pipeline_trace_t* fence_state_;
    uint64_t pending_loads_;
    bool fence_lock_;
};

///////////////////////////////////////////////////////////////////////////////

class AluUnit : public ExeUnit {
public:
    AluUnit(const SimContext& ctx, Core*);
    
    void tick();
};

///////////////////////////////////////////////////////////////////////////////

class CsrUnit : public ExeUnit {
public:
    CsrUnit(const SimContext& ctx, Core*);
    
    void tick();
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public ExeUnit {
public:
    FpuUnit(const SimContext& ctx, Core*);
    
    void tick();
};

///////////////////////////////////////////////////////////////////////////////

class GpuUnit : public ExeUnit {
public:
    GpuUnit(const SimContext& ctx, Core*);
    
    void tick();

private:    
  std::vector<RasterUnit::Ptr> raster_units_;  
  std::vector<RopUnit::Ptr>    rop_units_;
  std::vector<TexUnit::Ptr>    tex_units_;
  std::vector<SimPort<pipeline_trace_t*>*> pending_rsps_;
};

}