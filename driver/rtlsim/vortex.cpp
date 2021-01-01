#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <chrono>

#include <vortex.h>
#include <VX_config.h>
#include <ram.h>
#include <simulator.h>

///////////////////////////////////////////////////////////////////////////////

inline size_t align_size(size_t size, size_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

///////////////////////////////////////////////////////////////////////////////

class vx_device;

class vx_buffer {
public:
    vx_buffer(size_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        auto aligned_asize = align_size(size, CACHE_BLOCK_SIZE);
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

    size_t size() const {
        return size_;
    }

    vx_device* device() const {
        return device_;
    }

private:
    size_t size_;
    vx_device* device_;
    void* data_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {    
public:
    vx_device() {        
        mem_allocation_ = ALLOC_BASE_ADDR;        
    } 

    ~vx_device() {    
        if (future_.valid()) {
            future_.wait();
        }
    }

    int alloc_local_mem(size_t size, size_t* dev_maddr) {
        auto dev_mem_size = LOCAL_MEM_SIZE;
        size_t asize = align_size(size, CACHE_BLOCK_SIZE);        
        if (mem_allocation_ + asize > dev_mem_size)
            return -1;
        *dev_maddr = mem_allocation_;
        mem_allocation_ += asize;
        return 0;
    }

    int upload(void* src, size_t dest_addr, size_t size, size_t src_offset) {
        size_t asize = align_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > ram_.size())
            return -1;

        /*printf("VXDRV: upload %d bytes to 0x%x\n", size, dest_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-write: 0x%x <- 0x%x\n", uint32_t(dest_addr + i), *(uint32_t*)((uint8_t*)src + src_offset + i));
        }*/
        
        ram_.write(dest_addr, asize, (uint8_t*)src + src_offset);
        return 0;
    }

    int download(const void* dest, size_t src_addr, size_t size, size_t dest_offset) {
        size_t asize = align_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > ram_.size())
            return -1;

        ram_.read(src_addr, asize, (uint8_t*)dest + dest_offset);
        
        /*printf("VXDRV: download %d bytes from 0x%x\n", size, src_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", uint32_t(src_addr + i), *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int start() {   
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }
        simulator_.attach_ram(&ram_);
        future_ = std::async(std::launch::async, [&]{             
            simulator_.reset();        
            while (simulator_.is_busy()) {
                simulator_.step();
            }
        });
        return 0;
    }

    int wait(long long timeout) {
        if (!future_.valid())
            return 0;
        auto timeout_sec = (timeout < 0) ? timeout : (timeout / 1000);
        std::chrono::seconds wait_time(1);
        for (;;) {
            auto status = future_.wait_for(wait_time); // wait for 1 sec and check status
            if (status == std::future_status::ready 
             || 0 == timeout_sec--)
                break;
        }
        return 0;
    }

    int set_csr(int core_id, int addr, unsigned value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        simulator_.set_csr(core_id, addr, value);        
        while (simulator_.csr_req_active()) {
            simulator_.step();
        };
        return 0;
    }

    int get_csr(int core_id, int addr, unsigned *value) {
        if (future_.valid()) {
            future_.wait(); // ensure prior run completed
        }        
        simulator_.get_csr(core_id, addr, value);        
        while (simulator_.csr_req_active()) {
            simulator_.step();
        };
        return 0;
    }

private:

    size_t mem_allocation_;     
    RAM ram_;
    Simulator simulator_;
    std::future<void> future_;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, unsigned caps_id, unsigned *value) {
   if (nullptr == hdevice)
        return  -1;

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = IMPLEMENTATION_ID;
        break;
    case VX_CAPS_MAX_CORES:
        *value = NUM_CORES;        
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
        *value = 0xffffffff;
        break;
    case VX_CAPS_ALLOC_BASE_ADDR:
        *value = 0x10000000;
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

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    
#ifdef DUMP_PERF_STATS
    vx_dump_perf(device, stdout);
#endif

    delete device;

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->alloc_local_mem(size, dev_maddr);
}


extern int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer) {
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

extern int vx_buf_release(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    delete buffer;

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + src_offset > buffer->size())
        return -1;

    return buffer->device()->upload(buffer->data(), dev_maddr, size, src_offset);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dest_offset) {
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

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->wait(timeout);
}

extern int vx_csr_set(vx_device_h hdevice, int core_id, int addr, unsigned value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->set_csr(core_id, addr, value);
}

extern int vx_csr_get(vx_device_h hdevice, int core_id, int addr, unsigned* value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->get_csr(core_id, addr, value);
}