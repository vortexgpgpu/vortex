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

protected:
    Core* core_;
};

///////////////////////////////////////////////////////////////////////////////

class NopUnit : public ExeUnit {
public:
    NopUnit(const SimContext& ctx, Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public ExeUnit {
private:    
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_dcache_;
    pipeline_trace_t* fence_state_;
    bool fence_lock_;

public:
    LsuUnit(const SimContext& ctx, Core*);

    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class AluUnit : public ExeUnit {
public:
    AluUnit(const SimContext& ctx, Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class CsrUnit : public ExeUnit {
public:
    CsrUnit(const SimContext& ctx, Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public ExeUnit {
public:
    FpuUnit(const SimContext& ctx, Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class GpuUnit : public ExeUnit {
private:
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_tex_reqs_;

    bool processTexRequest(uint64_t cycle, pipeline_trace_t* trace);
    
public:
    GpuUnit(const SimContext& ctx, Core*);
    
    void step(uint64_t cycle);
};

}