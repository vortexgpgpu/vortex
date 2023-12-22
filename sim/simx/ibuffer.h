// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "pipeline.h"
#include <queue>

namespace vortex {

class IBuffer {
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

private:
    std::queue<pipeline_trace_t*> entries_;
    uint32_t capacity_;
};

}