#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

#include <vx_driver.h>

#include "../../simX/include/debug.h"
#include "../../simX/include/types.h"
#include "../../simX/include/core.h"
#include "../../simX/include/enc.h"
#include "../../simX/include/instruction.h"
#include "../../simX/include/mem.h"
#include "../../simX/include/obj.h"
#include "../../simX/include/archdef.h"
#include "../../simX/include/help.h"

#define CACHE_LINESIZE		64

#define PAGE_SIZE           4096

#define CHECK_RES(_expr)                                            \
   do {                                                             \
     fpga_result res = _expr;                                       \
     if (res == FPGA_OK)                                            \
       break;                                                       \
     printf("OPAE Error: '%s' returned %d!\n", #_expr, (int)res);   \
     return -1;                                                     \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

static size_t align_size(size_t size) {
    return CACHE_LINESIZE * ((size + CACHE_LINESIZE - 1) / CACHE_LINESIZE);
}

///////////////////////////////////////////////////////////////////////////////

class vx_device;

class vx_buffer {
public:
    vx_buffer(size_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        auto aligned_asize = align_size(size);
        data_ = malloc(aligned_asize);
    }

    ~vx_buffer() {
        if (data_) {
            free(data_);
        }
    }

    auto data() const {
        return data_;
    }

    auto size() const {
        return size_;
    }

    auto device() const {
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
        : is_done_(false)
        , is_running_(false)
        , thread_(__thread_proc__, this)
    {}

    ~vx_device() {
        mutex_.lock();
        is_done_ = true;
        mutex_.unlock();
        
        thread_.join();
    }

    int upload(void* src, size_t dest_addr, size_t size, size_t src_offset) {
        if (dest_addr + size > ram_.size())
            return -1;
        ram_.write(dest_addr, size, (uint8_t*)src + src_offset);
        return 0;
    }

    int download(const void* dest, size_t src_addr, size_t size, size_t dest_offset) {
        if (src_addr + size > ram_.size())
            return -1;
        ram_.read(src_addr, size, (uint8_t*)dest + dest_offset);
        return 0;
    }

    int start() {        
        if (this->wait(-1) != 0)
            return -1;

        mutex_.lock();     
        is_running_ = true;
        mutex_.unlock();

        return 0;
    }

    int wait(long long timeout) {
        for (;;) {
            mutex_.lock();
            bool is_running = is_running_;
            mutex_.unlock();

            if (!is_running || 0 == timeout--)
                break;

            std::this_thread::sleep_for(std::chrono::milliseconds(1));            
        }
        return 0;
    }

private:

    void run() {        
        Harp::ArchDef arch("rv32i", false);
        Harp::WordDecoder dec(arch);
        Harp::MemoryUnit mu(PAGE_SIZE, arch.getWordSize(), true);
        Harp::Core core(arch, dec, mu);
        mu.attach(ram_, 0);  

        while (core.running()) { 
            core.step();
        }
        core.printStats();
    }

    void thread_proc() {
        std::cout << "Device ready..." << std::endl;

        for (;;) {
            mutex_.lock();
            bool is_done = is_done_;
            bool is_running = is_running_;
            mutex_.unlock();

            if (is_done)
                break;

            if (is_running) {                                
                std::cout << "Device running..." << std::endl;
                
                this->run();

                mutex_.lock();
                is_running_ = false;
                mutex_.unlock();

                std::cout << "Device ready..." << std::endl;
            }
        }

        std::cout << "Device shutdown..." << std::endl;
    }

    static void __thread_proc__(vx_device* device) {
        device->thread_proc();
    }

    bool is_done_;
    bool is_running_;    
    std::thread thread_;   
    Harp::RAM ram_;
    std::mutex mutex_;
};

///////////////////////////////////////////////////////////////////////////////

extern vx_device_h vx_dev_open() {

    auto device = new vx_device();

    return (vx_device_h)device;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    delete (vx_device*)hdevice;

    return 0;
}

extern vx_buffer_h vx_buf_alloc(vx_device_h hdevice, size_t size) {
    if (nullptr == hdevice)
        return nullptr;

    auto buffer = new vx_buffer(size, (vx_device*)hdevice);
    if (nullptr == buffer->data()) {
        delete buffer;
        return nullptr;
    }

    return (vx_buffer*)buffer;
}

extern void* vs_buf_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    return ((vx_buffer*)hbuffer)->data();
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    delete (vx_buffer*)hbuffer;

    return 0;
}

extern int vx_copy_to_fpga(vx_buffer_h hbuffer, size_t dest_addr, size_t size, size_t src_offset) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + src_offset > buffer->size())
        return -1;

    return buffer->device()->upload(buffer->data(), dest_addr, size, src_offset);
}

extern int vx_copy_from_fpga(vx_buffer_h hbuffer, size_t src_addr, size_t size, size_t dest_offset) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + dest_offset > buffer->size())
        return -1;    

    return buffer->device()->download(buffer->data(), src_addr, size, dest_offset);
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    return ((vx_device*)hdevice)->start();
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (nullptr == hdevice)
        return -1;

    return ((vx_device*)hdevice)->wait(timeout);
}
