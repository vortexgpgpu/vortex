#include <vortex.h>
#include <vx_malloc.h>
#include <vx_utils.h>
#include <VX_config.h>
#include <VX_types.h>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#include "experimental/xrt_error.h"

#define RAM_PAGE_SIZE 4096

#define DEFAULT_DEVICE_INDEX 0

#define DEFAULT_XCLBIN_PATH "vortex_afu.xclbin"

#define KERNEL_NAME "vortex_afu"

#define CHECK_HANDLE(handle, _expr)                     \
    do {                                                \
        handle = _expr;                                 \
        if (handle !== 0)                               \
            break;                                      \
        printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr); \
        return -1;                                      \
    } while (false)

#define CHECK_XERR(_expr)                               \
    do {                                                \
        xrtErrorCode err = _expr;                       \
        if (err == 0)                                   \
            break;                                      \
        size_t len = 0;                                 \
        xrtErrorGetString(xrtDevice, err, nullptr, 0, &len); \
        std::vector<char> buf(len);                     \
        xrtErrorGetString(xrtDevice, err, buf.data(), buf.size(), nullptr); \
        printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err, buf.data()); \
        return -1;                                      \
    } while (false)

using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device(xrtDeviceHandle device, 
              xrtKernelHandle kernel)
        : xrtDevice(device)
        , xrtKernel(kernel)
    {}
    
    ~vx_device() {}

    xrtDeviceHandle xrtDevice;
    xrtKernelHandle xrtKernel;

    DeviceConfig dcrs;
    unsigned version;
    unsigned num_cores;
    unsigned num_warps;
    unsigned num_threads;
    uint64_t isa_caps;
};

///////////////////////////////////////////////////////////////////////////////

class vx_buffer {
public:
    vx_buffer(xrtBufferHandle buffer) : xrtBuffer(buffer) {}
    
    ~vx_buffer() {}

    xrtBufferHandle xrtBuffer;
};

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = device->version;
        break;
    case VX_CAPS_MAX_CORES:
        *value = device->num_cores;
        break;
    case VX_CAPS_MAX_WARPS:
        *value = device->num_warps;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = device->num_threads;
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
        *value = device->dcrs.read(DCR_BASE_STARTUP_ADDR);
        break;
    case VX_CAPS_ISA_FLAGS:
        *value = device->isa_caps;
        break;
    default:
        fprintf(stderr, "[VXDRV] Error: invalid caps id: %d\n", caps_id);
        std::abort();
        return -1;
    }

    return 0;
}

extern int vx_dev_open(vx_device_h* hdevice) {
    int err;

    if (nullptr == hdevice)
        return -1;

    int device_index = DEFAULT_DEVICE_INDEX;    
    const char* device_index_s = getenv("XRT_DEVICE_INDEX");
    if (device_index_s != nullptr) {
        device_index = atoi(device_index_s);
    }   

    const char* xlbin_path_s = getenv("XRT_XCLBIN_PATH");
    if (xlbin_path_s == nullptr) {
        xlbin_path_s = DEFAULT_XCLBIN_PATH;
    }   

    const char* emul_mode_s = getenv("XCL_EMULATION_MODE");
    if (emul_mode_s != nullptr) {
        printf("XCL_EMULATION_MODE=%s\n", emul_mode_s);
    }

    xrtDeviceHandle xrtDevice;
    xrtKernelHandle xrtKernel;

    xrtDevice = xrtDeviceOpen(device_index);
    if (nullptr == xrtDevice)
        return -1;

    printf("part1\n");

    err = xrtDeviceLoadXclbinFile(xrtDevice, xlbin_path_s);
    if (err != 0) {        
        xrtDeviceClose(xrtDevice);
        return -1;
    }

    printf("part2\n");

    xuid_t uuid;
    err = xrtDeviceGetXclbinUUID(xrtDevice, uuid);
    if (err != 0) {        
        xrtDeviceClose(xrtDevice);
        return -1;
    }

    printf("part3\n");

    xrtKernel = xrtPLKernelOpen(xrtDevice, uuid, KERNEL_NAME);
    if (nullptr == xrtKernel) {
        xrtDeviceClose(xrtDevice);
        return -1;
    }

    auto device = new vx_device(xrtDevice, xrtKernel);
    if (nullptr == device) {
        xrtKernelClose(xrtKernel);
        xrtDeviceClose(xrtDevice);
        return -1;
    }

    auto xrtBuffer2 = xrtBOAlloc(xrtDevice, 0x100000, 0, 1); 
    if (nullptr == xrtBuffer2) {
        return -1;
    }
    auto phyaddr2 = xrtBOAddress(xrtBuffer2);

    // 32 banks * 256 MB
    auto xrtBuffer1 = xrtBOAlloc(xrtDevice, 0x100000 * 256, 0, 0); 
    if (nullptr == xrtBuffer1) {
        return -1;
    
    }
    auto phyaddr1 = xrtBOAddress(xrtBuffer1);

    
        
    dcr_initialize(device);

#ifdef DUMP_PERF_STATS
    perf_add_device(device);
#endif 

    *hdevice = device;

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    xrtKernelClose(device->xrtKernel);
 
    xrtDeviceClose(device->xrtDevice);

    delete device;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    auto handle = xrtBOAlloc(device, size, XRT_BO_FLAGS_NONE, (xrtMemoryGroup) xrtKernelArgGroupId(device->xrtKernel, 4));

    return 0;
}

int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    return 0;
}

extern int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    auto handle = xrtBOAlloc(device, size, XRT_BO_FLAGS_NONE, (xrtMemoryGroup) xrtKernelArgGroupId(device->xrtKernel, 4));

    auto buffer = new vx_buffer(handle);

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    auto buffer = (vx_buffer*)hbuffer;

    return reinterpret_cast<void*>(xrtBOMap(buffer->xrtBuffer));
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    xrtBOFree(buffer->xrtBuffer);

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    //rtBOWrite(device_handle, host_ptr, size, device_offset);
    //xrtBOSync(device_handle, XCL_BO_SYNC_BO_TO_DEVICE, size, device_offset);
    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    //rtBOWrite(device_handle, host_ptr, size, device_offset);
    //xrtBOSync(device_handle, XCL_BO_SYNC_BO_TO_DEVICE, size, device_offset);
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    //auto run = xrtKernelRun(device->xrtKernel, s0, s1, s2, s3, axi00 != nullptr ? axi00 : dummy_buffers[0], axi01 != nullptr ? axi01 : dummy_buffers[1]);

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    /*while (true) { 
        auto res = xrtRunWait(run);
        if (res != ERT_CMD_STATE_COMPLETED)
            printf("%s: error: %llu\n", __FUNCTION__, res);
        else
            break;
    }

    xrtRunClose(run);*/

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    //xrtKernelWriteRegister(device->xrtKernel, addr, value);
    
    return 0;
}