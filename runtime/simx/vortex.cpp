#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <chrono>

#include <vortex.h>
#include <utils.h>
#include <malloc.h>

#include <VX_config.h>
#include <VX_types.h>

#include <util.h>

#include <processor.h>
#include <arch.h>
#include <mem.h>
#include <constants.h>

#ifndef NDEBUG
#define DBGPRINT(format, ...) do { printf("[VXDRV] " format "", ##__VA_ARGS__); } while (0)
#else
#define DBGPRINT(format, ...) ((void)0)
#endif

using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device;

class vx_buffer {
public:
    vx_buffer(uint64_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        uint64_t aligned_asize = aligned_size(size, CACHE_BLOCK_SIZE);
        data_ = aligned_malloc(aligned_asize, CACHE_BLOCK_SIZE);
        // set uninitialized data to "baadf00d"
        for (uint32_t i = 0; i < aligned_asize; ++i) {
            ((uint8_t*)data_)[i] = (0xbaadf00d >> ((i & 0x3) * 8)) & 0xff;
        }
    }

    ~vx_buffer() {
        if (data_) {
            aligned_free(data_);
        }
    }

    void* data() const {
        return data_;
    }

    uint64_t size() const {
        return size_;
    }

    vx_device* device() const {
        return device_;
    }

private:
    uint64_t   size_;
    vx_device* device_;
    void*      data_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {    
public:
    vx_device() 
        : arch_(NUM_THREADS, NUM_WARPS, NUM_CORES, NUM_CLUSTERS)
        , ram_(RAM_PAGE_SIZE)
        , processor_(arch_)
        , mem_allocator_(
            ALLOC_BASE_ADDR,
            ALLOC_MAX_ADDR,
            RAM_PAGE_SIZE,
            CACHE_BLOCK_SIZE) 
    {
        // attach memory module
        processor_.attach_ram(&ram_);
    }

    ~vx_device() {
        if (future_.valid()) {
            future_.wait();
        }
    }    

    int mem_alloc(uint64_t size, uint64_t* dev_maddr) {
        return mem_allocator_.allocate(size, dev_maddr);
    }

    int mem_free(uint64_t dev_maddr) {
        return mem_allocator_.release(dev_maddr);
    }

    uint64_t mem_used() const {
        return mem_allocator_.allocated();
    }

    int upload(const void* src, uint64_t dest_addr, uint64_t size, uint64_t src_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        ram_.write((const uint8_t*)src + src_offset, dest_addr, asize);
        
        /*DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
        for (uint64_t i = 0; i < size && i < 1024; i += 4) {
            DBGPRINT("  0x%lx <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + src_offset + i));
        }*/
        
        return 0;
    }

    int download(void* dest, uint64_t src_addr, uint64_t size, uint64_t dest_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        ram_.read((uint8_t*)dest + dest_offset, src_addr, asize);
        
        /*DBGPRINT("download %ld bytes from 0x%lx\n", size, src_addr);
        for (uint64_t i = 0; i < size && i < 1024; i += 4) {
            DBGPRINT("  0x%lx -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int start() {  
        // ensure prior run completed
        if (future_.valid()) {
            future_.wait();
        }
        
        // start new run
        future_ = std::async(std::launch::async, [&]{
            processor_.run(false);
        });
        
        return 0;
    }

    int wait(uint64_t timeout) {
        if (!future_.valid())
            return 0;
        uint64_t timeout_sec = timeout / 1000;
        std::chrono::seconds wait_time(1);
        for (;;) {
            // wait for 1 sec and check status
            auto status = future_.wait_for(wait_time);
            if (status == std::future_status::ready 
             || 0 == timeout_sec--)
                break;
        }
        return 0;
    } 

    int write_dcr(uint32_t addr, uint32_t value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        processor_.write_dcr(addr, value);
        dcrs_.write(addr, value);
        return 0;
    }

    uint64_t read_dcr(uint32_t addr) const {
        return dcrs_.read(addr);
    }

private:
    Arch                arch_;
    RAM                 ram_;
    Processor           processor_;
    MemoryAllocator     mem_allocator_;
    DeviceConfig        dcrs_;
    std::future<void>   future_;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return  -1;

    auto device = new vx_device();
    if (device == nullptr)
        return -1;

    int err = dcr_initialize(device);
    if (err != 0) {
        delete device;
        return err;
    }

#ifdef DUMP_PERF_STATS
    perf_add_device(device);
#endif  

    *hdevice = device;

    DBGPRINT("device creation complete!\n");

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

#ifdef DUMP_PERF_STATS
    perf_remove_device(hdevice);
#endif

    delete device;

    DBGPRINT("device destroyed!\n");

    return 0;
}

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return  -1;

    vx_device *device = ((vx_device*)hdevice);

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = IMPLEMENTATION_ID;
        break;
    case VX_CAPS_NUM_THREADS:
        *value = NUM_THREADS;
        break;
    case VX_CAPS_NUM_WARPS:
        *value = NUM_WARPS;
        break;
    case VX_CAPS_NUM_CORES:
        *value = NUM_CORES;
        break;
    case VX_CAPS_NUM_CLUSTERS:
        *value = NUM_CLUSTERS;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = LOCAL_MEM_SIZE;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = (uint64_t(device->read_dcr(DCR_BASE_STARTUP_ADDR1)) << 32) | 
                device->read_dcr(DCR_BASE_STARTUP_ADDR0);
        break;    
    case VX_CAPS_ISA_FLAGS:
        *value = ((uint64_t(MISA_EXT))<<32) | ((log2floor(XLEN)-4) << 30) | MISA_STD;
        break;
    default:
        std::cout << "invalid caps id: " << caps_id << std::endl;
        std::abort();
        return -1;
    }

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 == size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->mem_alloc(size, dev_maddr);
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    if (0 == dev_maddr)
        return 0;

    vx_device *device = ((vx_device*)hdevice);
    return device->mem_free(dev_maddr);
}

extern int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_total) {
    if (nullptr == hdevice)
        return -1;

    auto device = ((vx_device*)hdevice);
    if (mem_free) {
        *mem_free = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR) - device->mem_used();
    }
    if (mem_total) {
        *mem_total = (ALLOC_MAX_ADDR - ALLOC_BASE_ADDR);
    }
    return 0;
}

extern int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice 
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    auto buffer = new vx_buffer(size, device);
    if (nullptr == buffer->data()) {
        delete buffer;
        return -1;
    }

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    return buffer->data();
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    delete buffer;

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + src_offset > buffer->size())
        return -1;

    DBGPRINT("COPY_TO_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld, src_offset=%ld\n", 
        dev_maddr, (uintptr_t)buffer->data(), size, src_offset);

    return buffer->device()->upload(buffer->data(), dev_maddr, size, src_offset);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    if (nullptr == hbuffer 
      || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;
    if (size + dest_offset > buffer->size())
        return -1;

    DBGPRINT("COPY_FROM_DEV: dev_addr=0x%lx, host_addr=0x%lx, size=%ld, dest_offset=0x%lx\n", 
        dev_maddr, (uintptr_t)buffer->data(), size, dest_offset); 

    return buffer->device()->download(buffer->data(), dev_maddr, size, dest_offset);
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;    
    
    DBGPRINT("START\n");

    vx_device *device = ((vx_device*)hdevice);

    return device->start();
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->wait(timeout);
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;

    DBGPRINT("DCR_WRITE: addr=0x%x, value=0x%lx\n", addr, value);
  
    return device->write_dcr(addr, value);
}