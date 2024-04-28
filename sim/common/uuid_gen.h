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

    uint32_t get_uuid(uint64_t /*PC*/) {
        /*uint16_t id;
        uint16_t ref;
        auto it = uuid_map_.find(PC);
        if (it != uuid_map_.end()) {
            uint32_t value = it->second;
            ref = value & 0xffff;
            id  = value >> 16;
            ++ref;
        } else {
            ref = 0;
            id = ids_++;
        }
        uint32_t ret = (uint32_t(id) << 16) | ref;
        uuid_map_[PC] = ret;*/
        return ids_++;
    }

    void reset() {
        //uuid_map_.clear();
        ids_ = 0;
    }

private:

    //std::unordered_map<uint64_t, uint32_t> uuid_map_;
    uint16_t ids_;
};

}