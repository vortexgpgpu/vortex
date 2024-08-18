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

#include "instr_trace.h"
#include <queue>

namespace vortex {

class PipelineLatch {
public:
  PipelineLatch() {}
  ~PipelineLatch() {}
  
  bool empty() const {
    return queue_.empty();
  }

  instr_trace_t* front() {
    return queue_.front();
  }

  void push(instr_trace_t* value) {    
    queue_.push(value);
  }

  void pop() {
    queue_.pop();
  }

  void clear() {
    std::queue<instr_trace_t*> empty;
    std::swap(queue_, empty);
  }

protected:
  std::queue<instr_trace_t*> queue_;
};

}