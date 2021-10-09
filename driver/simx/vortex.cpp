#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

#include <vortex.h>
#include <core.h>
#include <VX_config.h>
#include <util.h>

#define PAGE_SIZE   4096

using namespace vortex;

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
    vx_device() 
        : arch_("rv32i", NUM_CORES, NUM_WARPS, NUM_THREADS)
        , decoder_(arch_)
        , mmu_(PAGE_SIZE, arch_.wsize(), true)
        , cores_(arch_.num_cores())
        , is_done_(false)
        , is_running_(false)
        , thread_(__thread_proc__, this)
        , ram_((1<<12), (1<<20))  {

        mem_allocation_ = ALLOC_BASE_ADDR;               
        mmu_.attach(ram_, 0, 0xffffffff);  
        for (int i = 0; i < arch_.num_cores(); ++i) {
            cores_[i] = std::make_shared<Core>(arch_, decoder_, mmu_, i);
        }
    }

    ~vx_device() {
        mutex_.lock();
        is_done_ = true;
        mutex_.unlock();
        
        thread_.join();
    }

    int alloc_local_mem(size_t size, size_t* dev_maddr) {
        auto dev_mem_size = LOCAL_MEM_SIZE;
        auto asize = align_size(size, CACHE_BLOCK_SIZE);        
        if (mem_allocation_ + asize > dev_mem_size)
            return -1;
        *dev_maddr = mem_allocation_;
        mem_allocation_ += asize;
        return 0;
    }

    int upload(const void* src, size_t dest_addr, size_t size, size_t src_offset) {
        auto asize = align_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > ram_.size())
            return -1;

        ram_.write((const uint8_t*)src + src_offset, dest_addr, asize);
        
        /*printf("VXDRV: upload %d bytes to 0x%x\n", size, dest_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-write: 0x%x <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + src_offset + i));
        }*/
        
        return 0;
    }

    int download(void* dest, size_t src_addr, size_t size, size_t dest_offset) {
        size_t asize = align_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > ram_.size())
            return -1;

        ram_.read((uint8_t*)dest + dest_offset, src_addr, asize);
        
        /*printf("VXDRV: download %d bytes from 0x%x\n", size, src_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int start() {  

        mutex_.lock();     
        for (int i = 0; i < arch_.num_cores(); ++i) {
            cores_[i]->clear();
        }
        is_running_ = true;        
        mutex_.unlock();

        return 0;
    }

    int wait(long long timeout) {
        auto timeout_sec = (timeout < 0) ? timeout : (timeout / 1000);
        for (;;) {
            mutex_.lock();
            bool is_running = is_running_;
            mutex_.unlock();

            if (!is_running || 0 == timeout_sec--)
                break;

            std::this_thread::sleep_for(std::chrono::seconds(1));            
        }
        return 0;
    }

    int get_csr(int core_id, int addr, unsigned *value) {
        *value = cores_.at(core_id)->get_csr(addr, 0, 0);
        return 0;
    }    

    int set_csr(int core_id, int addr, unsigned value) {
        cores_.at(core_id)->set_csr(addr, value, 0, 0);
        return 0;
    }

private:

    void run() {
        bool running;
        do {
            running = false;
            for (auto& core : cores_) {
                core->step();
                if (core->running())
                    running = true;
            }
        } while (running);
    }

    void thread_proc() {
        std::cout << "Device ready..." << std::flush << std::endl;

        for (;;) {
            mutex_.lock();
            bool is_done = is_done_;
            bool is_running = is_running_;
            mutex_.unlock();

            if (is_done)
                break;

            if (is_running) {                                
                std::cout << "Device running..." << std::flush << std::endl;
                
                this->run();

                mutex_.lock();
                is_running_ = false;
                mutex_.unlock();

                std::cout << "Device ready..." << std::flush << std::endl;
            }
        }

        std::cout << "Device shutdown..." << std::flush << std::endl;
    }

    static void __thread_proc__(vx_device* device) {
        device->thread_proc();
    }

    ArchDef arch_;
    Decoder decoder_;
    MemoryUnit mmu_;
    std::vector<std::shared_ptr<Core>> cores_;
    bool is_done_;
    bool is_running_;   
    size_t mem_allocation_; 
    std::thread thread_;   
    RAM ram_;
    std::mutex mutex_;
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

extern int vx_dev_caps(vx_device_h hdevice, unsigned caps_id, unsigned *value) {
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