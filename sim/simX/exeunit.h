#pragma once

#include <simobject.h>
#include "pipeline.h"
#include "cache.h"

namespace vortex {

class Core;

class ExeUnit {
protected:
    const char* name_;
    Queue<pipeline_trace_t*> inputs_;
    Queue<pipeline_trace_t*> outputs_;

    void schedule_output(pipeline_trace_t* trace, uint32_t delay) {
        if (delay > 1) {
            SimPlatform::instance().schedule(
                [&](pipeline_trace_t* req) { 
                    outputs_.push(req); 
                },
                trace,
                (delay - 1)
            );
        } else {
            outputs_.push(trace);
        }
    }

public:    
    typedef std::shared_ptr<ExeUnit> Ptr;

    ExeUnit(const char* name) : name_(name) {}    
    virtual ~ExeUnit() {}

    void push(pipeline_trace_t* trace) {
        inputs_.push(trace);
    }

    bool empty() const {
        return outputs_.empty();
    }

    pipeline_trace_t* top() const {
        return outputs_.top();
    }

    void pop() {
        outputs_.pop();
    }

    virtual void step(uint64_t cycle) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class NopUnit : public ExeUnit {
public:
    NopUnit(Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public ExeUnit {
private:
    Core* core_;
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_dcache_;
    pipeline_trace_t* fence_state_;
    bool fence_lock_;

public:
    LsuUnit(Core*);

    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class AluUnit : public ExeUnit {
public:
    AluUnit(Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class CsrUnit : public ExeUnit {
public:
    CsrUnit(Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public ExeUnit {
public:
    FpuUnit(Core*);
    
    void step(uint64_t cycle);
};

///////////////////////////////////////////////////////////////////////////////

class GpuUnit : public ExeUnit {
private:
    Core* core_;
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_trace_t*, uint32_t>> pending_tex_reqs_;

    bool processTexRequest(uint64_t cycle, pipeline_trace_t* trace);
    
public:
    GpuUnit(Core*);
    
    void step(uint64_t cycle);
};

}