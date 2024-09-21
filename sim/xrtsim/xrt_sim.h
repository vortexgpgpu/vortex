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

#include <stdint.h>

namespace vortex {

class xrt_sim {
public:

  xrt_sim();
  virtual ~xrt_sim();

  int init();

  int mem_alloc(uint64_t size, uint32_t bank_id, uint64_t* addr);

  int mem_free(uint32_t bank_id, uint64_t addr);

  int mem_write(uint32_t bank_id, uint64_t addr, uint64_t size, const void* value);

  int mem_read(uint32_t bank_id, uint64_t addr, uint64_t size, void* value);

  int register_write(uint32_t offset, uint32_t value);

  int register_read(uint32_t offset, uint32_t* value);

private:

  class Impl;
  Impl* impl_;
};

}