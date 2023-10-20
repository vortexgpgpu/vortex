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

#include <unordered_map>

namespace vortex {

class UUIDGenerator {
public:    
    UUIDGenerator() : ids_(0) {}
    virtual ~UUIDGenerator() {}

    uint32_t get_uuid(uint64_t PC) {
        uint32_t id;
        uint32_t ref;
        auto it = uuid_map_.find(PC);
        if (it != uuid_map_.end()) {
            uint64_t value = it->second;
            id  = value & 0xffff;
            ref = value >> 16;
        } else {
            id = ids_++;
            ref = -1;
        }
        ++ref;
        uint64_t ret = (uint64_t(ref) << 16) | id;
        uuid_map_[PC] = ret;
        return ret;
    }

    void reset() {
        uuid_map_.clear();
        ids_ = 0;
    }

private:

    std::unordered_map<uint64_t, uint32_t> uuid_map_;
    uint32_t ids_;
};

}