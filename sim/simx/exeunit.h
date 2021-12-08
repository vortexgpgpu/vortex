#pragma once

#include <simobject.h>
#include "pipeline.h"
#include "cache.h"

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

class NopUnit : public ExeUnit {
public:
    NopUnit(const SimContext& ctx, Core*);
    
    void tick();
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public ExeUnit {
private:    
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_rd_reqs_;
    pipeline_trace_t* fence_state_;
    bool fence_lock_;

public:
    LsuUnit(const SimContext& ctx, Core*);

    void reset();

    void tick();
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
private:
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_tex_reqs_;

    bool processTexRequest(pipeline_trace_t* trace);
    
public:
    GpuUnit(const SimContext& ctx, Core*);

    void reset();
    
    void tick();
};

}