#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class IBuffer {
private:
    std::queue<pipeline_trace_t*> entries_;
    uint32_t capacity_;

public:    
    IBuffer(uint32_t size) 
        : capacity_(size)
    {}

    bool empty() const {
        return entries_.empty();
    }
    
    bool full() const {
        return (entries_.size() == capacity_);
    }

    pipeline_trace_t* top() const {
        return entries_.front();
    }

    void push(pipeline_trace_t* trace) {
        entries_.emplace(trace);
    }

    void pop() {
        return entries_.pop();
    }

    void clear() {
        std::queue<pipeline_trace_t*> empty;
        std::swap(entries_, empty );
    }
};

}