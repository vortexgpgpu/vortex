#pragma once

#include <simobject.h>
#include "pipeline.h"
#include "cache.h"

namespace vortex {

class Core;

class ExeUnit {
protected:
    const char* name_;
    Queue<pipeline_state_t> inputs_;
    Queue<pipeline_state_t> outputs_;

    void schedule_output(const pipeline_state_t& state, uint32_t delay) {
        if (delay > 1) {
            SimPlatform::instance().schedule(
                [&](const pipeline_state_t& req) { 
                    outputs_.push(req); 
                },
                state,
                (delay - 1)
            );
        } else {
            outputs_.push(state);
        }
    }

public:    
    typedef std::shared_ptr<ExeUnit> Ptr;

    ExeUnit(const char* name) : name_(name) {}
    
    virtual ~ExeUnit() {}

    void push_input(const pipeline_state_t& state) {
        inputs_.push(state);
    }

    bool pop_output(pipeline_state_t* state) {
        return outputs_.try_pop(state);
    }

    virtual void step() = 0;
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public ExeUnit {
private:
    Core* core_;
    uint32_t num_threads_;
    HashTable<std::pair<pipeline_state_t, ThreadMask>> pending_dcache_;
    pipeline_state_t fence_state_;
    bool fence_lock_;

public:
    LsuUnit(Core*);

    void handleCacheReponse(const MemRsp& response, uint32_t port_id);

    void step();
};

///////////////////////////////////////////////////////////////////////////////

class AluUnit : public ExeUnit {
public:
    AluUnit(Core*);
    
    void step();
};

///////////////////////////////////////////////////////////////////////////////

class CsrUnit : public ExeUnit {
public:
    CsrUnit(Core*);
    
    void step();
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public ExeUnit {
public:
    FpuUnit(Core*);
    
    void step();
};

///////////////////////////////////////////////////////////////////////////////

class GpuUnit : public ExeUnit {
public:
    GpuUnit(Core*);
    
    void step();
};

}