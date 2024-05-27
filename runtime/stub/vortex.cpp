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

#include <callbacks.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

static callbacks_t g_callbacks;
static void* g_drv_handle = nullptr;

typedef int (*vx_dev_init_t)(callbacks_t*);

int vx_dev_open(vx_device_h* hdevice) {
    {
        const char* driverName = getenv("VORTEX_DRIVER");
        if (driverName == nullptr) {
            driverName = "simx";
        }
        std::string driverName_s(driverName);
        std::string libName = "libvortex-" + driverName_s + ".so";
        auto handle = dlopen(libName.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            std::cerr << "Cannot open library: " << dlerror() << std::endl;
            return 1;
        }

        auto vx_dev_init = (vx_dev_init_t)dlsym(handle, "vx_dev_init");
        auto dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "Cannot load symbol 'vx_init': " << dlsym_error << std::endl;
            dlclose(handle);
            return 1;
        }

        vx_dev_init(&g_callbacks);
        g_drv_handle = handle;
    }

    return (g_callbacks.dev_open)(hdevice);
}

int vx_dev_close(vx_device_h hdevice) {
    int ret = (g_callbacks.dev_close)(hdevice);
    dlclose(g_drv_handle);
    return ret;
}

int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t* value) {
    return (g_callbacks.dev_caps)(hdevice, caps_id, value);
}

int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer) {
    return (g_callbacks.mem_alloc)(hdevice, size, flags, hbuffer);
}

int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer) {
    return (g_callbacks.mem_reserve)(hdevice, address, size, flags, hbuffer);
}

int vx_mem_free(vx_buffer_h hbuffer) {
    return (g_callbacks.mem_free)(hbuffer);
}

int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags) {
    return (g_callbacks.mem_access)(hbuffer, offset, size, flags);
}

int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address) {
    return (g_callbacks.mem_address)(hbuffer, address);
}

int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used) {
    return (g_callbacks.mem_info)(hdevice, mem_free, mem_used);
}

int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size) {
    return (g_callbacks.copy_to_dev)(hbuffer, host_ptr, dst_offset, size);
}

int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size) {
    return (g_callbacks.copy_from_dev)(host_ptr, hbuffer, src_offset, size);
}

int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments) {
    return (g_callbacks.start)(hdevice, hkernel, harguments);
}

int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    return (g_callbacks.ready_wait)(hdevice, timeout);
}

int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t* value) {
    return (g_callbacks.dcr_read)(hdevice, addr, value);
}

int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value) {
    return (g_callbacks.dcr_write)(hdevice, addr, value);
}

int vx_mpm_query(vx_device_h hdevice, uint32_t addr, uint32_t core_id, uint64_t* value) {
    return (g_callbacks.mpm_query)(hdevice, addr, core_id, value);
}