#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

#include <vortex.h>
#include <ram.h>

#ifdef USE_MULTICORE
#include <Vortex_SOC.h>
#else
#include <Vortex.h>
#endif

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
    return VX_CACHE_LINESIZE * ((size + VX_CACHE_LINESIZE - 1) / VX_CACHE_LINESIZE);
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
        : is_done_(false)
        , mem_allocation_(VX_ALLOC_BASE_ADDR)
        , vortex_(&ram_) {
        thread_ = new std::thread(__thread_proc__, this);        
    } 

    ~vx_device() {        
        if (thread_) {
            mutex_.lock();
            is_done_ = true;
            mutex_.unlock();

            thread_->join();
            delete thread_;
        }
    }

    int alloc_local_mem(size_t size, size_t* dev_maddr) {
        size_t asize = align_size(size);
        if (mem_allocation_ + asize > VX_LOCAL_MEM_SIZE)
            return -1;
        *dev_maddr = mem_allocation_;
        mem_allocation_ += asize;
        return 0;
    }

    int upload(void* src, size_t dest_addr, size_t size, size_t src_offset) {
        size_t asize = align_size(size);
        if (dest_addr + asize > ram_.size())
            return -1;

        /*printf("VXDRV: upload %d bytes to 0x%x\n", size, dest_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-write: 0x%x <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + src_offset + i));
        }*/
        
        ram_.write(dest_addr, asize, (uint8_t*)src + src_offset);
        return 0;
    }

    int download(const void* dest, size_t src_addr, size_t size, size_t dest_offset) {
        size_t asize = align_size(size);
        if (src_addr + asize > ram_.size())
            return -1;

        ram_.read(src_addr, asize, (uint8_t*)dest + dest_offset);
        
        /*printf("VXDRV: download %d bytes from 0x%x\n", size, src_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int flush_caches(size_t dev_maddr, size_t size) {
        
        mutex_.lock();     
        vortex_.flush_caches(dev_maddr, size);
        mutex_.unlock();

        return 0;
    }

    int start() {   

        mutex_.lock();     
        vortex_.reset();
        mutex_.unlock();

        return 0;
    }

    int wait(long long timeout) {
        auto timeout_sec = (timeout < 0) ? timeout : (timeout / 1000);
        for (;;) {
            mutex_.lock();
            bool is_busy = vortex_.is_busy();
            mutex_.unlock();

            if (!is_busy || 0 == timeout_sec--)
                break;

            std::this_thread::sleep_for(std::chrono::seconds(1));            
        }
        return 0;
    }

private:

    void thread_proc() {
        std::cout << "Device ready..." << std::endl;

        for (;;) {
            mutex_.lock();
            bool is_done = is_done_;
            mutex_.unlock();

            if (is_done)
                break;

            mutex_.lock();
            vortex_.step();
            mutex_.unlock();
        }

        std::cout << "Device shutdown..." << std::endl;
    }

    static void __thread_proc__(vx_device* device) {
        device->thread_proc();
    }

    bool is_done_; 
    size_t mem_allocation_;     
    RAM ram_;
#ifdef USE_MULTICORE
    Vortex_SOC vortex_;
#else
    Vortex vortex_;
#endif
    std::thread* thread_;   
    std::mutex mutex_;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    if (NULL == hdevice)
        return  -1;

    *hdevice = new vx_device();

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    delete device;

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (NULL == hdevice 
     || NULL == dev_maddr
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->alloc_local_mem(size, dev_maddr);
}

extern int vx_flush_caches(vx_device_h hdevice, size_t dev_maddr, size_t size) {
    if (NULL == hdevice 
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->flush_caches(dev_maddr, size);
}


extern int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice 
     || 0 >= size
     || NULL == hbuffer)
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

extern volatile void* vx_host_ptr(vx_buffer_h hbuffer) {
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
