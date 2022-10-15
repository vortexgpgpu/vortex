#pragma once

#include <vortex.h>
#include <fpga.h>
#include "vx_utils.h"
#include "vx_malloc.h"

#define CHECK_HANDLE(handle, _expr, _cleanup)   \
    auto handle = _expr;                        \
    if (handle == nullptr) {                    \
        printf("[VXDRV] Error: '%s' returned NULL!\n", #_expr); \
        _cleanup                                \
    }

#define CHECK_ERR(_expr, _cleanup)              \
    do {                                        \
        auto err = _expr;                       \
        if (err == 0)                           \
            break;                              \
        printf("[VXDRV] Error: '%s' returned %d, %s!\n", #_expr, (int)err, api.fpgaErrStr(err)); \
        _cleanup                                \
    } while (false)

typedef fpga_result (*pfn_fpgaGetProperties)(fpga_token token, fpga_properties *prop);
typedef fpga_result (*pfn_fpgaPropertiesSetObjectType)(fpga_properties prop, fpga_objtype objtype);
typedef fpga_result (*pfn_fpgaPropertiesSetGUID)(fpga_properties prop, fpga_guid guid);
typedef fpga_result (*pfn_fpgaDestroyProperties)(fpga_properties *prop);
typedef fpga_result (*pfn_fpgaEnumerate)(const fpga_properties *filters, uint32_t num_filters, fpga_token *tokens, uint32_t max_tokens, uint32_t *num_matches);
typedef fpga_result (*pfn_fpgaDestroyToken)(fpga_token *token);

typedef fpga_result (*pfn_fpgaOpen)(fpga_token token, fpga_handle *handle, int flags);
typedef fpga_result (*pfn_fpgaClose)(fpga_handle handle);
typedef fpga_result (*pfn_fpgaPrepareBuffer)(fpga_handle handle, uint64_t len, void **buf_addr, uint64_t *wsid, int flags);
typedef fpga_result (*pfn_fpgaReleaseBuffer)(fpga_handle handle, uint64_t wsid);
typedef fpga_result (*pfn_fpgaGetIOAddress)(fpga_handle handle, uint64_t wsid, uint64_t *ioaddr);
typedef fpga_result (*pfn_fpgaWriteMMIO64)(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t value);
typedef fpga_result (*pfn_fpgaReadMMIO64)(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t *value);
typedef const char *(*pfn_fpgaErrStr)(fpga_result e);

struct opae_api_funcs_t {
	pfn_fpgaGetProperties 	fpgaGetProperties;
	pfn_fpgaPropertiesSetObjectType fpgaPropertiesSetObjectType;
	pfn_fpgaPropertiesSetGUID fpgaPropertiesSetGUID;
	pfn_fpgaDestroyProperties fpgaDestroyProperties;
	pfn_fpgaEnumerate 		fpgaEnumerate;
	pfn_fpgaDestroyToken 	fpgaDestroyToken;

	pfn_fpgaOpen 			fpgaOpen;
	pfn_fpgaClose 			fpgaClose;
	pfn_fpgaPrepareBuffer 	fpgaPrepareBuffer;
	pfn_fpgaReleaseBuffer 	fpgaReleaseBuffer;
	pfn_fpgaGetIOAddress 	fpgaGetIOAddress;
	pfn_fpgaWriteMMIO64  	fpgaWriteMMIO64;
	pfn_fpgaReadMMIO64     	fpgaReadMMIO64;
	pfn_fpgaErrStr        	fpgaErrStr;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device() 
        : mem_allocator(
            ALLOC_BASE_ADDR, 
            ALLOC_BASE_ADDR + LOCAL_MEM_SIZE,
            4096,            
            CACHE_BLOCK_SIZE)
    {}
    
    ~vx_device() {}

    opae_api_funcs_t api;

    fpga_handle fpga;
    vortex::MemoryAllocator mem_allocator;    
    DeviceConfig dcrs;
    unsigned version;
    unsigned num_cores;
    unsigned num_warps;
    unsigned num_threads;
    uint64_t isa_caps;
};

///////////////////////////////////////////////////////////////////////////////

typedef struct vx_buffer_ {
    uint64_t wsid;
    void* host_ptr;
    uint64_t io_addr;
    vx_device_h hdevice;
    uint64_t size;
} vx_buffer_t;

///////////////////////////////////////////////////////////////////////////////

int api_init(opae_api_funcs_t* opae_api_funcs);