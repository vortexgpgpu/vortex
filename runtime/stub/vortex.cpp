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

#include <vortex.h>

extern int vx_dev_open(vx_device_h* /*hdevice*/) {
    return -1;
}

extern int vx_dev_close(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_dev_caps(vx_device_h /*hdevice*/, uint32_t /*caps_id*/, uint64_t* /*value*/) {
    return -1;
}

extern int vx_mem_alloc(vx_device_h /*hdevice*/, uint64_t /*size*/, int /*type*/, uint64_t* /*dev_addr*/) {
    return -1;
}

extern int vx_mem_free(vx_device_h /*hdevice*/, uint64_t /*dev_addr*/) {
    return -1;
}

extern int vx_mem_info(vx_device_h /*hdevice*/, int /*type*/, uint64_t* /*mem_free*/, uint64_t* /*mem_used*/) {
    return 0;
}

extern int vx_copy_to_dev(vx_device_h /*hdevice*/, uint64_t /*dev_addr*/, const void* /*host_ptr*/, uint64_t /*size*/) {
    return -1;
}

extern int vx_copy_from_dev(vx_device_h /*hdevice*/, void* /*host_ptr*/, uint64_t /*dev_addr*/, uint64_t /*size*/) {
     return -1;
}

extern int vx_start(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_ready_wait(vx_device_h /*hdevice*/, uint64_t /*timeout*/) {
    return -1;
}

extern int vx_dcr_write(vx_device_h /*hdevice*/, uint32_t /*addr*/, uint64_t /*value*/) {
    return -1;
}
