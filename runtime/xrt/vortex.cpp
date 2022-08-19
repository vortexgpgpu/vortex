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

#define RAM_PAGE_SIZE 4096

#define DEFAULT_DEVICE_INDEX 0

#define DEFAULT_XCLBIN_PATH "vortex_kernel.xclbin"

#define KERNEL_NAME "vortex_kernel"

using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device(xrtDeviceHandle xrtdevice_, 
              xrtXclbinHandle xclbin_,
              xrtKernelHandle xrtkernel_)
        : xrtdevice(xrtdevice_)
        , xclbin(xclbin_)
        , xrtkernel(xrtkernel_)
    {}
    
    ~vx_device() {}

    xrtDeviceHandle xrtdevice;
    xrtXclbinHandle xclbin;
    xrtKernelHandle xrtkernel;

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
    vx_buffer(vx_device& vx_device, size_t size) : bo(device, size, mem_used.get_index()) 
    {

    }
    
    ~vx_buffer() {}

    xrt::bo bo;

    uint64_t wsid;
    void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    uint64_t size;
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

    xrtDeviceHandle xrtdevice;
    xrtXclbinHandle xclbin;
    xrtKernelHandle xrtkernel;

    xrtdevice = xrtDeviceOpen(device_index);
    if (!xrtdevice)
        return -1;

    xclbin = xrtXclbinAllocFilename(xlbin_path_s);
    if (!xclbin) {
        xrtDeviceClose(xrtdevice);
        return -1;
    }

    xrtDeviceLoadXclbinHandle(xrtdevice, xclbin);

    xuid_t uuid;
    xrtXclbinGetUUID(xclbin, uuid);

    xrtkernel = xrtPLKernelOpen(xrtdevice, uuid, KERNEL_NAME);
    if (!xrtkernel) {
        return -1;
    }

    auto device = new vx_device(xrtdevice, xclbin, xrtKernelHandle);
    if (nullptr == device)        
        return -1;    
        
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

    xrtXclbinFreeHandle(device->xclbin);

    xrtKernelClose(device->xrtkernel);
 
    xrtDeviceClose(device->xrtdevice);

    delete device;

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    device_handle = xrtBOAlloc(device, size, XRT_BO_FLAGS_NONE, (xrtMemoryGroup) xrtKernelArgGroupId(kernel, 4)));

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

    auto handle = xrtBOAlloc(device, size, XRT_BO_FLAGS_NONE, (xrtMemoryGroup) xrtKernelArgGroupId(kernel, 4)));

    auto buffer = new vx_buffer(handle);

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    return reinterpret_cast<void*>(xrtBOMap(buffer->xrtbo));
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    xrtBOFree(buffer->xrtbo);

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    rtBOWrite(device_handle, host_ptr, size, device_offset);
    xrtBOSync(device_handle, XCL_BO_SYNC_BO_TO_DEVICE, size, device_offset);
    return 0;
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
    rtBOWrite(device_handle, host_ptr, size, device_offset);
    xrtBOSync(device_handle, XCL_BO_SYNC_BO_TO_DEVICE, size, device_offset);
    return 0;
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    auto run = xrtKernelRun(kernel, s0, s1, s2, s3, axi00 != NULL ? axi00 : dummy_buffers[0], axi01 != NULL ? axi01 : dummy_buffers[1]);

    return 0;
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    while (true) { 
        res = xrtRunWait(run);
        if (res != ERT_CMD_STATE_COMPLETED)
            printf("%s: error: %llu\n", __FUNCTION__, res);
        else
            break;
    }

    xrtRunClose(run);

    return 0;
}

extern int vx_dcr_write(vx_device_h hdevice, uint32_t addr, uint64_t value) {
    if (nullptr == hdevice)
        return -1;

    auto device = (vx_device*)hdevice;

    xrtKernelWriteRegister(device->xrtkernel, addr, value);
    
    return 0;
}