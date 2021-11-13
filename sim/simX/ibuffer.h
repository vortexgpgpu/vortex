#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class IBuffer {
private:
    std::queue<pipeline_state_t> entries_;
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

    const pipeline_state_t& top() const {
        return entries_.front();
    }

    void push(const pipeline_state_t& state) {
        entries_.emplace(state);
    }

    void pop() {
        return entries_.pop();
    }
};

}