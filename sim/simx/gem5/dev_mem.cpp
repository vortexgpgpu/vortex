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

#include "dev_mem.h"

#include <mem.h>

namespace vortex_gem5 {

void InProcessDevMem::read(uint64_t addr, void* dst, std::size_t bytes) {
    ram_.enable_acl(false);
    ram_.read(static_cast<uint8_t*>(dst), addr, bytes);
    ram_.enable_acl(true);
}

void InProcessDevMem::write(uint64_t addr, const void* src, std::size_t bytes) {
    ram_.enable_acl(false);
    ram_.write(static_cast<const uint8_t*>(src), addr, bytes);
    ram_.enable_acl(true);
}

} // namespace vortex_gem5
