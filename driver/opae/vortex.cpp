#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <cmath>
#include <thread>
#include <future>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <uuid/uuid.h>
#include <opae/fpga.h>
#include <vortex.h>
#include "vortex_afu.h"

#define CHECK_RES(_expr)                                            \
   do {                                                             \
     fpga_result res = _expr;                                       \
     if (res == FPGA_OK)                                            \
       break;                                                       \
     printf("OPAE Error: '%s' returned %d, %s!\n",                  \
            #_expr, (int)res, fpgaErrStr(res));                     \
     return -1;                                                     \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

#define CMD_TYPE_READ           AFU_IMAGE_CMD_TYPE_READ
#define CMD_TYPE_WRITE          AFU_IMAGE_CMD_TYPE_WRITE
#define CMD_TYPE_RUN            AFU_IMAGE_CMD_TYPE_RUN
#define CMD_TYPE_CLFLUSH        AFU_IMAGE_CMD_TYPE_CLFLUSH

#define MMIO_CSR_CMD            (AFU_IMAGE_MMIO_CSR_CMD * 4)
#define MMIO_CSR_IO_ADDR        (AFU_IMAGE_MMIO_CSR_IO_ADDR * 4)
#define MMIO_CSR_MEM_ADDR       (AFU_IMAGE_MMIO_CSR_MEM_ADDR * 4)
#define MMIO_CSR_DATA_SIZE      (AFU_IMAGE_MMIO_CSR_DATA_SIZE * 4)
#define MMIO_CSR_STATUS         (AFU_IMAGE_MMIO_CSR_STATUS * 4)
#define MMIO_CSR_SCOPE_CMD      (AFU_IMAGE_MMIO_CSR_SCOPE_CMD * 4)
#define MMIO_CSR_SCOPE_DATA     (AFU_IMAGE_MMIO_CSR_SCOPE_DATA * 4)

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_device_ {
    fpga_handle fpga;
    size_t mem_allocation;
} vx_device_t;

typedef struct vx_buffer_ {
    uint64_t wsid;
    volatile void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    size_t size;
} vx_buffer_t;

inline size_t align_size(size_t size, size_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

inline bool is_aligned(size_t addr, size_t alignment) {
    assert(0 == (alignment & (alignment - 1)));
    return 0 == (addr & (alignment - 1));
}

///////////////////////////////////////////////////////////////////////////////

static int vx_scope_trace(vx_device_h hdevice) {      
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    std::ofstream ofs("vx_scope.vcd");

    ofs << "$timescale 1 ns $end" << std::endl;

    int fwidth = 0;

    ofs << "$var reg 1  0 clk $end" << std::endl;

    fwidth += 1;

    ofs << "$var reg 1  1 icache_req_valid $end" << std::endl;
    ofs << "$var reg 1  2 icache_req_ready $end" << std::endl;
    ofs << "$var reg 1  3 icache_rsp_valid $end" << std::endl;
    ofs << "$var reg 1  4 icache_rsp_ready $end" << std::endl;
    ofs << "$var reg 4  5 dcache_req_valid $end" << std::endl;  
    ofs << "$var reg 1  6 dcache_req_ready $end" << std::endl; 
    ofs << "$var reg 4  7 dcache_rsp_valid $end" << std::endl; 
    ofs << "$var reg 1  8 dcache_rsp_ready $end" << std::endl;
    ofs << "$var reg 1  9 dram_req_valid $end" << std::endl;   
    ofs << "$var reg 1 10 dram_req_ready $end" << std::endl;
    ofs << "$var reg 1 11 dram_rsp_valid $end" << std::endl;
    ofs << "$var reg 1 12 dram_rsp_ready $end" << std::endl;
    ofs << "$var reg 1 13 schedule_delay $end" << std::endl;

    fwidth += 19;

    ofs << "$var reg 2  14 icache_req_tag $end" << std::endl;    
    ofs << "$var reg 2  15 icache_rsp_tag $end" << std::endl;
    ofs << "$var reg 2  16 dcache_req_tag $end" << std::endl;
    ofs << "$var reg 2  17 dcache_rsp_tag $end" << std::endl;     
    ofs << "$var reg 29 18 dram_req_tag $end" << std::endl;
    ofs << "$var reg 29 19 dram_rsp_tag $end" << std::endl;

    fwidth += 66;

    const int num_signals = 20;

    ofs << "enddefinitions $end" << std::endl;

    uint64_t frame_width, max_frames, data_valid;

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 2));
    CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &frame_width));
    std::cout << "scope::frame_width=" << frame_width << std::endl;

    assert((fwidth-1)== (int)frame_width);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 3));
    CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &max_frames));
    std::cout << "scope::max_frames=" << max_frames << std::endl;

    do {
        CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 0));
        CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));        
        if (data_valid)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (true);

    std::cout << "scope trace dump begin..." << std::endl;

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 1));

    std::vector<char> signa_data(frame_width+1);

    uint64_t frame_offset = 0, frame_no = 0, timestamp = 0;
    
    int signal_id = 0;
    int signal_offset = 0;

    auto print_header = [&] () {
        ofs << '#' << timestamp++ << std::endl;
        ofs << "b0 0" << std::endl;
        ofs << '#' << timestamp++ << std::endl;
        ofs << "b1 0" << std::endl;
        
        uint64_t delta;
        fpga_result res = fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &delta);
        assert(res == FPGA_OK);

        while (delta != 0) {
            ofs << '#' << timestamp++ << std::endl;
            ofs << "b0 0" << std::endl;
            ofs << '#' << timestamp++ << std::endl;
            ofs << "b1 0" << std::endl;
            --delta;
        }

        signal_id = 1;
    };
    
    auto print_signal = [&] (uint64_t word, int signal_width) {

        int word_offset = frame_offset % 64;

        signa_data[signal_width - signal_offset - 1] = ((word >> word_offset) & 0x1) ? '1' : '0';

        ++signal_offset;
        ++frame_offset;

        if (signal_offset == signal_width) {
            signa_data[signal_width] = 0; // string null termination
            ofs << 'b' << signa_data.data() << ' ' << (num_signals - signal_id) << std::endl;
            signal_offset = 0;            
            ++signal_id;
        }

        if (frame_offset == frame_width) {   
            assert(0 == signal_offset);                     
            signal_id = 0;
            frame_offset = 0;
            ++frame_no;
            if (frame_no != max_frames) {                
                print_header();
            }                        
        }
    };

    print_header();

    do {
        if (frame_no == max_frames-1) {
            // verify last frame is valid
            CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 0));
            CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));  
            assert(data_valid == 1);
            CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 1));
        }

        uint64_t word;
        CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &word));
        
        do {
            switch (num_signals - signal_id) {
            case 14:
            case 15:
            case 16:
            case 17:
                print_signal(word, 2); 
                break;
            case 5:
            case 7:
                print_signal(word, 4); 
                break;
            case 18:
            case 19: 
                print_signal(word, 29); 
                break;
            default: 
                print_signal(word, 1);
                break;
            }           
        } while ((frame_offset % 64) != 0);

    } while (frame_no != max_frames);

    std::cout << "scope trace dump done!" << std::endl;

    // verify data not valid
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_CMD, 0));
    CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_SCOPE_DATA, &data_valid));  
    assert(data_valid == 0);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    fpga_properties filter = nullptr;
    fpga_result res;
    fpga_guid guid;
    fpga_token accel_token;
    uint32_t num_matches;
    fpga_handle accel_handle;
    vx_device_t* device;

    if (nullptr == hdevice)
        return  -1;

    // ensure that the block size 64
    assert(64 == vx_dev_caps(VX_CAPS_CACHE_LINESIZE));

    // Set up a filter that will search for an accelerator
    fpgaGetProperties(nullptr, &filter);
    fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);

    // Add the desired UUID to the filter
    uuid_parse(AFU_ACCEL_UUID, guid);
    fpgaPropertiesSetGUID(filter, guid);

    // Do the search across the available FPGA contexts
    num_matches = 1;
    fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);

    // Not needed anymore
    fpgaDestroyProperties(&filter);

    if (num_matches < 1) {
        fprintf(stderr, "Accelerator %s not found!\n", AFU_ACCEL_UUID);
        return -1;
    }

    // Open accelerator
    res = fpgaOpen(accel_token, &accel_handle, 0);
    if (FPGA_OK != res) {
        return -1;
    }

    // Done with token
    fpgaDestroyToken(&accel_token);

    // allocate device object
    device = (vx_device_t*)malloc(sizeof(vx_device_t));
    if (nullptr == device) {
        fpgaClose(accel_handle);
        return -1;
    }

    device->fpga = accel_handle;
    device->mem_allocation = vx_dev_caps(VX_CAPS_ALLOC_BASE_ADDR);

    *hdevice = device;

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    fpgaClose(device->fpga);

    free(device);

    return 0;
}

extern int vx_alloc_dev_mem(vx_device_h hdevice, size_t size, size_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    int line_size = vx_dev_caps(VX_CAPS_CACHE_LINESIZE);    
    size_t dev_mem_size = vx_dev_caps(VX_CAPS_LOCAL_MEM_SIZE);

    size_t asize = align_size(size, line_size);
    
    if (device->mem_allocation + asize > dev_mem_size)
        return -1;   

    *dev_maddr = device->mem_allocation;
    device->mem_allocation += asize;

    return 0;
}

extern int vx_alloc_shared_mem(vx_device_h hdevice, size_t size, vx_buffer_h* hbuffer) {
    fpga_result res;
    void* host_ptr;
    uint64_t wsid;
    uint64_t io_addr;
    vx_buffer_t* buffer;

    if (nullptr == hdevice
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    int line_size = vx_dev_caps(VX_CAPS_CACHE_LINESIZE);

    size_t asize = align_size(size, line_size);

    res = fpgaPrepareBuffer(device->fpga, asize, &host_ptr, &wsid, 0);
    if (FPGA_OK != res) {
        return -1;
    }

    // Get the physical address of the buffer in the accelerator
    res = fpgaGetIOAddress(device->fpga, wsid, &io_addr);
    if (FPGA_OK != res) {
        fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    }

    // allocate buffer object
    buffer = (vx_buffer_t*)malloc(sizeof(vx_buffer_t));
    if (nullptr == buffer) {
        fpgaReleaseBuffer(device->fpga, wsid);
        return -1;
    }

    buffer->wsid     = wsid;
    buffer->host_ptr = host_ptr;
    buffer->io_addr  = io_addr;
    buffer->hdevice  = hdevice;
    buffer->size     = asize;

    *hbuffer = buffer;

    return 0;
}

extern volatile void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);

    return buffer->host_ptr;
}

extern int vx_buf_release(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer_t* buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    fpgaReleaseBuffer(device->fpga, buffer->wsid);

    free(buffer);

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, long long timeout) {
    if (nullptr == hdevice)
        return -1;
    
    vx_device_t *device = ((vx_device_t*)hdevice);

    uint64_t data = 0;
    struct timespec sleep_time; 

#if defined(USE_ASE)
    sleep_time.tv_sec = 1;
    sleep_time.tv_nsec = 0;
#else
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 1000000;
#endif

    // to milliseconds
    long long sleep_time_ms = (sleep_time.tv_sec * 1000) + (sleep_time.tv_nsec / 1000000);
    
    for (;;) {
        CHECK_RES(fpgaReadMMIO64(device->fpga, 0, MMIO_CSR_STATUS, &data));
        if (0 == data || 0 == timeout) {
            if (data != 0) {
                fprintf(stdout, "ready-wait timed out: status=%ld\n", data);
            }
            break;
        }
        nanosleep(&sleep_time, nullptr);
        timeout -= sleep_time_ms;
    };

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t *buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    int line_size = vx_dev_caps(VX_CAPS_CACHE_LINESIZE);   
    size_t dev_mem_size = vx_dev_caps(VX_CAPS_LOCAL_MEM_SIZE); 

    size_t asize = align_size(size, line_size);

    // check alignment
    if (!is_aligned(dev_maddr, line_size))
        return -1;
    if (!is_aligned(buffer->io_addr + src_offset, line_size))
        return -1;
    
    // bound checking
    if (src_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    auto ls_shift = (int)std::log2(line_size);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_IO_ADDR, (buffer->io_addr + src_offset) >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_MEM_ADDR, (dev_maddr >> ls_shift) ));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_DATA_SIZE, asize >> ls_shift));   
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CMD, CMD_TYPE_WRITE));

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, size_t dev_maddr, size_t size, size_t dest_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    vx_buffer_t *buffer = ((vx_buffer_t*)hbuffer);
    vx_device_t *device = ((vx_device_t*)buffer->hdevice);

    int line_size = vx_dev_caps(VX_CAPS_CACHE_LINESIZE); 
    size_t dev_mem_size = vx_dev_caps(VX_CAPS_LOCAL_MEM_SIZE);  

    size_t asize = align_size(size, line_size);

    // check alignment
    if (!is_aligned(dev_maddr, line_size))
        return -1;
    if (!is_aligned(buffer->io_addr + dest_offset, line_size))
        return -1; 

    // bound checking
    if (dest_offset + asize > buffer->size)
        return -1;
    if (dev_maddr + asize > dev_mem_size)
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    auto ls_shift = (int)std::log2(line_size);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_IO_ADDR, (buffer->io_addr + dest_offset) >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_MEM_ADDR, (dev_maddr) >> ls_shift));    
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_DATA_SIZE, asize >> ls_shift));   
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CMD, CMD_TYPE_READ));

    // Wait for the write operation to finish
    if (vx_ready_wait(buffer->hdevice, -1) != 0)
        return -1;

    return 0;
}

extern int vx_flush_caches(vx_device_h hdevice, size_t dev_maddr, size_t size) {
    if (nullptr == hdevice 
     || 0 >= size)
        return -1;

    vx_device_t* device = ((vx_device_t*)hdevice);

    int line_size = vx_dev_caps(VX_CAPS_CACHE_LINESIZE); 

    size_t asize = align_size(size, line_size);  

    // check alignment
    if (!is_aligned(dev_maddr, line_size))
        return -1;

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;

    auto ls_shift = (int)std::log2(line_size);

    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_MEM_ADDR, dev_maddr >> ls_shift));
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_DATA_SIZE, asize >> ls_shift));   
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CMD, CMD_TYPE_CLFLUSH));

    // Wait for the write operation to finish
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;

    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device_t *device = ((vx_device_t*)hdevice);

    // Ensure ready for new command
    if (vx_ready_wait(hdevice, -1) != 0)
        return -1;    

    // start execution
    CHECK_RES(fpgaWriteMMIO64(device->fpga, 0, MMIO_CSR_CMD, CMD_TYPE_RUN));

#ifdef SCOPE
    vx_scope_trace(hdevice);
#endif

    return 0;
}