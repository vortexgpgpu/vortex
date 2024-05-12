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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <list>
#include <chrono>

#include <vortex.h>
#include <malloc.h>
#include <utils.h>
#include <VX_config.h>
#include <VX_types.h>

#include <mem.h>
#include <util.h>
#include <processor.h>

using namespace vortex;

#define RAM_PAGE_SIZE 4096

#ifndef NDEBUG
#define DBGPRINT(format, ...) do { printf("[VXDRV] " format "", ##__VA_ARGS__); } while (0)
#else
#define DBGPRINT(format, ...) ((void)0)
#endif

#define CHECK_ERR(_expr, _cleanup)              \
    do {                                        \
        auto err = _expr;                       \
        if (err == 0)                           \
            break;                              \
        printf("[VXDRV] Error: '%s' returned %d!\n", #_expr, (int)err); \
        _cleanup                                \
    } while (false)

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device()
        : ram_(0, RAM_PAGE_SIZE)
        , global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, RAM_PAGE_SIZE, CACHE_BLOCK_SIZE)
    {
        processor_.attach_ram(&ram_);
    }

    ~vx_device() {
        if (future_.valid()) {
            future_.wait();
        }
        profiling_remove(profiling_id_);
    }

    int init() {
        CHECK_ERR(dcr_initialize(this), {
            return err;
        });
        profiling_id_ = profiling_add(this);
        return 0;
    }

    int get_caps(uint32_t caps_id, uint64_t *value) {
        uint64_t _value;
        switch (caps_id) {
        case VX_CAPS_VERSION:
            _value = IMPLEMENTATION_ID;
            break;
        case VX_CAPS_NUM_THREADS:
            _value = NUM_THREADS;
            break;
        case VX_CAPS_NUM_WARPS:
            _value = NUM_WARPS;
            break;
        case VX_CAPS_NUM_CORES:
            _value = NUM_CORES * NUM_CLUSTERS;
            break;
        case VX_CAPS_NUM_BARRIERS:
            _value = NUM_BARRIERS;
            break;
        case VX_CAPS_CACHE_LINE_SIZE:
            _value = CACHE_BLOCK_SIZE;
            break;
        case VX_CAPS_GLOBAL_MEM_SIZE:
            _value = GLOBAL_MEM_SIZE;
            break;
        case VX_CAPS_LOCAL_MEM_SIZE:
            _value = (1 << LMEM_LOG_SIZE);
            break;
        case VX_CAPS_LOCAL_MEM_ADDR:
            _value = LMEM_BASE_ADDR;
            break;
        case VX_CAPS_ISA_FLAGS:
            _value = ((uint64_t(MISA_EXT))<<32) | ((log2floor(XLEN)-4) << 30) | MISA_STD;
            break;
        default:
            std::cout << "invalid caps id: " << caps_id << std::endl;
            std::abort();
            return -1;
        }
        *value = _value;
        return 0;
    }

    int mem_alloc(uint64_t size, int flags, uint64_t* dev_addr) {
        uint64_t addr;
        CHECK_ERR(global_mem_.allocate(size, &addr), {
            return err;
        });
        CHECK_ERR(this->mem_access(addr, size, flags), {
            global_mem_.release(addr);
            return err;
        });
        *dev_addr = addr;
        return 0;
    }

    int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
        CHECK_ERR(global_mem_.reserve(dev_addr, size), {
            return err;
        });
        CHECK_ERR(this->mem_access(dev_addr, size, flags), {
            global_mem_.release(dev_addr);
            return err;
        });
        return 0;
    }

    int mem_free(uint64_t dev_addr) {
        return global_mem_.release(dev_addr);
    }

    int mem_access(uint64_t dev_addr, uint64_t size, int flags) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dev_addr + asize > GLOBAL_MEM_SIZE)
            return -1;

        ram_.set_acl(dev_addr, size, flags);

        return 0;
    }

    int mem_info(uint64_t* mem_free, uint64_t* mem_used) const {
        if (mem_free)
            *mem_free = global_mem_.free();
        if (mem_used)
            *mem_used = global_mem_.allocated();
        return 0;
    }

    int upload(uint64_t dest_addr, const void* src, uint64_t size) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > GLOBAL_MEM_SIZE)
            return -1;

        ram_.enable_acl(false);
        ram_.write((const uint8_t*)src, dest_addr, size);
        ram_.enable_acl(true);

        /*printf("VXDRV: upload %ld bytes from 0x%lx:", size, uintptr_t((uint8_t*)src));
        for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
            printf("\n0x%08lx=", dest_addr + i * CACHE_BLOCK_SIZE);
            for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
                printf("%02x", *((uint8_t*)src + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
            }
        }
        printf("\n");*/

        return 0;
    }

    int download(void* dest, uint64_t src_addr, uint64_t size) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > GLOBAL_MEM_SIZE)
            return -1;

        ram_.enable_acl(false);
        ram_.read((uint8_t*)dest, src_addr, size);
        ram_.enable_acl(true);

        /*printf("VXDRV: download %ld bytes to 0x%lx:", size, uintptr_t((uint8_t*)dest));
        for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
            printf("\n0x%08lx=", src_addr + i * CACHE_BLOCK_SIZE);
            for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
                printf("%02x", *((uint8_t*)dest + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
            }
        }
        printf("\n");*/

        return 0;
    }

    int start(uint64_t krnl_addr, uint64_t args_addr) {
        // ensure prior run completed
        if (future_.valid()) {
            future_.wait();
        }

        // set kernel info
        this->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, krnl_addr & 0xffffffff);
        this->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, krnl_addr >> 32);
        this->dcr_write(VX_DCR_BASE_STARTUP_ARG0, args_addr & 0xffffffff);
        this->dcr_write(VX_DCR_BASE_STARTUP_ARG1, args_addr >> 32);

        profiling_begin(profiling_id_);

        // start new run
        future_ = std::async(std::launch::async, [&]{
            processor_.run();
        });

        // clear mpm cache
        mpm_cache_.clear();

        return 0;
    }

    int ready_wait(uint64_t timeout) {
        if (!future_.valid())
            return 0;
        uint64_t timeout_sec = timeout / 1000;
        std::chrono::seconds wait_time(1);
        for (;;) {
            // wait for 1 sec and check status
            auto status = future_.wait_for(wait_time);
            if (status == std::future_status::ready)
                break;
            if (0 == timeout_sec--)
                return -1;
        }
        profiling_end(profiling_id_);
        return 0;
    }

    int dcr_write(uint32_t addr, uint32_t value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }
        processor_.dcr_write(addr, value);
        dcrs_.write(addr, value);
        return 0;
    }

    int dcr_read(uint32_t addr, uint32_t* value) const {
        return dcrs_.read(addr, value);
    }

    int mpm_query(uint32_t addr, uint32_t core_id, uint64_t* value) {
        uint32_t offset = addr - VX_CSR_MPM_BASE;
        if (offset > 31)
            return -1;
        if (mpm_cache_.count(core_id) == 0) {
            uint64_t mpm_mem_addr = IO_MPM_ADDR + core_id * 32 * sizeof(uint64_t);
            CHECK_ERR(this->download(mpm_cache_[core_id].data(), mpm_mem_addr, 32 * sizeof(uint64_t)), {
                return err;
            });
        }
        *value = mpm_cache_.at(core_id).at(offset);
        return 0;
    }

private:

    RAM                 ram_;
    Processor           processor_;
    MemoryAllocator     global_mem_;
    DeviceConfig        dcrs_;
    std::future<void>   future_;
    std::unordered_map<uint32_t, std::array<uint64_t, 32>> mpm_cache_;
    int                 profiling_id_;
};

struct vx_buffer {
    vx_device* device;
    uint64_t addr;
    uint64_t size;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return  -1;

    auto device = new vx_device();
    if (device == nullptr)
        return -1;

    CHECK_ERR(device->init(), {
        delete device;
        return err;
    });

    DBGPRINT("DEV_OPEN: hdevice=%p\n", (void*)device);

    *hdevice = device;

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("DEV_CLOSE: hdevice=%p\n", hdevice);

    auto device = ((vx_device*)hdevice);

    delete device;

    return 0;
}

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
   if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    uint64_t _value;

    CHECK_ERR(device->get_caps(caps_id, &_value), {
        return err;
    });

    DBGPRINT("DEV_CAPS: hdevice=%p, caps_id=%d, value=%ld\n", hdevice, caps_id, _value);

    *value = _value;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice
     || nullptr == hbuffer
     || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint64_t dev_addr;
    CHECK_ERR(device->mem_alloc(size, flags, &dev_addr), {
        return err;
    });

    auto buffer = new vx_buffer{device, dev_addr, size};
    if (nullptr == buffer) {
        device->mem_free(dev_addr);
        return -1;
    }

    DBGPRINT("MEM_ALLOC: hdevice=%p, size=%ld, flags=0x%d, hbuffer=%p\n", hdevice, size, flags, (void*)buffer);

    *hbuffer = buffer;

    return 0;
}

extern int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice
     || nullptr == hbuffer
     || 0 == size)
        return -1;

    auto device = ((vx_device*)hdevice);

    CHECK_ERR(device->mem_reserve(address, size, flags), {
        return err;
    });

    auto buffer = new vx_buffer{device, address, size};
    if (nullptr == buffer) {
        device->mem_free(address);
        return -1;
    }

    DBGPRINT("MEM_RESERVE: hdevice=%p, address=0x%lx, size=%ld, flags=0x%d, hbuffer=%p\n", hdevice, address, size, flags, (void*)buffer);

    *hbuffer = buffer;

    return 0;
}

extern int vx_mem_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return 0;

    DBGPRINT("MEM_FREE: hbuffer=%p\n", hbuffer);

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    vx_mem_access(hbuffer, 0, buffer->size, 0);

    int err = device->mem_free(buffer->addr);

    delete buffer;

    return err;
}

extern int vx_mem_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags) {
   if (nullptr == hbuffer)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((offset + size) > buffer->size)
        return -1;

    DBGPRINT("MEM_ACCESS: hbuffer=%p, offset=%ld, size=%ld, flags=%d\n", hbuffer, offset, size, flags);

    return device->mem_access(buffer->addr + offset, size, flags);
}

extern int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);

    DBGPRINT("MEM_ADDRESS: hbuffer=%p, address=0x%lx\n", hbuffer, buffer->addr);

    *address = buffer->addr;

    return 0;
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint64_t _mem_free, _mem_used;

    CHECK_ERR(device->mem_info(&_mem_free, &_mem_used), {
        return err;
    });

    DBGPRINT("MEM_INFO: hdevice=%p, mem_free=%ld, mem_used=%ld\n", hdevice, _mem_free, _mem_used);

    if (mem_free) {
        *mem_free = _mem_free;
    }

    if (mem_used) {
        *mem_used = _mem_used;
    }

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size) {
    if (nullptr == hbuffer || nullptr == host_ptr)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((dst_offset + size) > buffer->size)
        return -1;

    DBGPRINT("COPY_TO_DEV: hbuffer=%p, host_addr=%p, dst_offset=%ld, size=%ld\n", hbuffer, host_ptr, dst_offset, size);

    return device->upload(buffer->addr + dst_offset, host_ptr, size);
}

extern int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size) {
    if (nullptr == hbuffer || nullptr == host_ptr)
        return -1;

    auto buffer = ((vx_buffer*)hbuffer);
    auto device = ((vx_device*)buffer->device);

    if ((src_offset + size) > buffer->size)
        return -1;

    DBGPRINT("COPY_FROM_DEV: hbuffer=%p, host_addr=%p, src_offset=%ld, size=%ld\n", hbuffer, host_ptr, src_offset, size);

    return device->download(host_ptr, buffer->addr + src_offset, size);
}

extern int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments) {
    if (nullptr == hdevice || nullptr == hkernel || nullptr == harguments)
        return -1;

    DBGPRINT("START: hdevice=%p, hkernel=%p, harguments=%p\n", hdevice, hkernel, harguments);

    auto device = ((vx_device*)hdevice);
    auto kernel = ((vx_buffer*)hkernel);
    auto arguments = ((vx_buffer*)harguments);

    return device->start(kernel->addr, arguments->addr);
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("READY_WAIT: hdevice=%p, timeout=%ld\n", hdevice, timeout);

    auto device = ((vx_device*)hdevice);

    return device->ready_wait(timeout);
}

extern int vx_dcr_read(vx_device_h hdevice, uint32_t addr, uint32_t* value) {
    if (nullptr == hdevice || NULL == value)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint32_t _value;

    CHECK_ERR(device->dcr_read(addr, &_value), {
        return err;
    });

    DBGPRINT("DCR_READ: hdevice=%p, addr=0x%x, value=0x%x\n", hdevice, addr, _value);

    *value = _value;

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint32_t value) {
    if (nullptr == hdevice)
        return -1;

    DBGPRINT("DCR_WRITE: hdevice=%p, addr=0x%x, value=0x%x\n", hdevice, addr, value);

    auto device = ((vx_device*)hdevice);

    return device->dcr_write(addr, value);
}

extern int vx_mpm_query(vx_device_h hdevice, uint32_t addr, uint32_t core_id, uint64_t* value) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);

    uint64_t _value;

    CHECK_ERR(device->mpm_query(addr, core_id, &_value), {
        return err;
    });

    DBGPRINT("MPM_QUERY: hdevice=%p, addr=0x%x, core_id=%d, value=0x%lx\n", hdevice, addr, core_id, _value);

    *value = _value;

    return 0;
}