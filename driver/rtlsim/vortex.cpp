#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <list>
#include <chrono>

#include <vortex.h>
#include <vx_malloc.h>
#include <vx_utils.h>
#include <VX_config.h>
#include <mem.h>
#include <util.h>
#include <processor.h>

#define RAM_PAGE_SIZE 4096

using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device;
class vx_buffer {
public:
    vx_buffer(uint64_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        auto aligned_asize = aligned_size(size, CACHE_BLOCK_SIZE);
        data_ = malloc(aligned_asize);
    }

    ~vx_buffer() {
        if (data_) {
            free(data_);
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
    uint64_t size_;
    vx_device* device_;
    void* data_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {    
public:
    vx_device() 
        : ram_(RAM_PAGE_SIZE)
        , mem_allocator_(
            ALLOC_BASE_ADDR,
            ALLOC_BASE_ADDR + LOCAL_MEM_SIZE,
            RAM_PAGE_SIZE,
            CACHE_BLOCK_SIZE) 
    {
        processor_.attach_ram(&ram_);
    }

    ~vx_device() {    
        if (future_.valid()) {
            future_.wait();
        }
    }

    int alloc_local_mem(uint64_t size, uint64_t* dev_maddr) {
        return mem_allocator_.allocate(size, dev_maddr);
    }

    int free_local_mem(uint64_t dev_maddr) {
        return mem_allocator_.release(dev_maddr);
    }

    int upload(const void* src, uint64_t dest_addr, uint64_t size, uint64_t src_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        /*printf("VXDRV: upload %ld bytes from 0x%lx:", size, uintptr_t((uint8_t*)src + src_offset));
        for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
            printf("\n0x%08lx=", dest_addr + i * CACHE_BLOCK_SIZE);
            for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
                printf("%02x", *((uint8_t*)src + src_offset + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
            }
        }
        printf("\n");*/
        
        ram_.write((const uint8_t*)src + src_offset, dest_addr, asize);
        return 0;
    }

    int download(void* dest, uint64_t src_addr, uint64_t size, uint64_t dest_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        ram_.read((uint8_t*)dest + dest_offset, src_addr, asize);
        
        /*printf("VXDRV: download %ld bytes to 0x%lx:", size, uintptr_t((uint8_t*)dest + dest_offset));
        for (int i = 0;  i < (asize / CACHE_BLOCK_SIZE); ++i) {
            printf("\n0x%08lx=", src_addr + i * CACHE_BLOCK_SIZE);
            for (int j = 0;  j < CACHE_BLOCK_SIZE; ++j) {
                printf("%02x", *((uint8_t*)dest + dest_offset + i * CACHE_BLOCK_SIZE + CACHE_BLOCK_SIZE - 1 - j));
            }
        }
        printf("\n");*/
        
        return 0;
    }

    int start() {   
        // ensure prior run completed
        if (future_.valid()) {
            future_.wait();
        }
        // start new run
        future_ = std::async(std::launch::async, [&]{
            processor_.run();
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

private:

    RAM ram_;
    Processor processor_;
    MemoryAllocator mem_allocator_;     
    std::future<void> future_;
};

///////////////////////////////////////////////////////////////////////////////

#ifdef DUMP_PERF_STATS
class AutoPerfDump {
private:
    std::list<vx_device_h> devices_;

public:
    AutoPerfDump() {} 

    ~AutoPerfDump() {
        for (auto device : devices_) {
            vx_dump_perf(device, stdout);
        }
    }

    void add_device(vx_device_h device) {
        devices_.push_back(device);
    }

    void remove_device(vx_device_h device) {
        devices_.remove(device);
    }    
};

AutoPerfDump gAutoPerfDump;
#endif

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
   if (nullptr == hdevice)
        return  -1;

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = IMPLEMENTATION_ID;
        break;
    case VX_CAPS_MAX_CORES:
        *value = NUM_CORES * NUM_CLUSTERS;        
        break;
    case VX_CAPS_MAX_WARPS:
        *value = NUM_WARPS;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = NUM_THREADS;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = LOCAL_MEM_SIZE;
        break;
    case VX_CAPS_ALLOC_BASE_ADDR:
        *value = ALLOC_BASE_ADDR;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = STARTUP_ADDR;
        break;
    default:
        std::cout << "invalid caps id: " << caps_id << std::endl;
        std::abort();
        return -1;
    }

    return 0;
}

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return  -1;

    *hdevice = new vx_device();

#ifdef DUMP_PERF_STATS
    gAutoPerfDump.add_device(*hdevice);
#endif

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    
#ifdef DUMP_PERF_STATS
    gAutoPerfDump.remove_device(hdevice);
    vx_dump_perf(hdevice, stdout);
#endif

    delete device;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->alloc_local_mem(size, dev_maddr);
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->free_local_mem(dev_maddr);
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

    return buffer->device()->upload(buffer->data(), dev_maddr, size, src_offset);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
     if (nullptr == hbuffer 
      || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + dest_offset > buffer->size())
        return -1;    

    return buffer->device()->download(buffer->data(), dev_maddr, size, dest_offset);
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->start();
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->wait(timeout);
}