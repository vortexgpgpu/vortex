#include <vortex.h>

extern int vx_dev_open(vx_device_h* /*hdevice*/) {
    return -1;
}

extern int vx_dev_close(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_dev_caps(vx_device_h /*hdevice*/, uint32_t /*caps_id*/, uint64_t* /*value*/) {
    return -1;
}

extern int vx_mem_alloc(vx_device_h /*hdevice*/, uint64_t /*size*/, uint64_t* /*dev_maddr*/) {
    return -1;
}

int vx_mem_free(vx_device_h /*hdevice*/, uint64_t /*dev_maddr*/) {
    return -1;
}

extern int vx_buf_alloc(vx_device_h /*hdevice*/, uint64_t /*size*/, vx_buffer_h* /*hbuffer*/) {
    return -1;
}

extern void* vx_host_ptr(vx_buffer_h /*hbuffer*/) {
    return nullptr;
}

extern int vx_buf_free(vx_buffer_h /*hbuffer*/) {
    return -1;
}

extern cmdbuffer* vx_create_command_buffer(uint64_t /*buf_size*/) {
    return nullptr;
}

extern int vx_new_copy_to_dev(vx_buffer_h /*hbuffer*/, uint64_t /*dev_maddr*/, uint64_t /*size*/, uint64_t /*src_offset*/, cmdbuffer* /*cmdBuf*/, uint64_t /*cmd_type*/) {
    return -1;
}

extern int vx_flush(cmdbuffer* /*cmdBuf*/) {
    return -1;
}

extern int vx_copy_to_dev(vx_buffer_h /*hbuffer*/, uint64_t /*dev_maddr*/, uint64_t /*size*/, uint64_t /*src_offset*/) {
    return -1;
}

extern int vx_copy_from_dev(vx_buffer_h /*hbuffer*/, uint64_t /*dev_maddr*/, uint64_t /*size*/, uint64_t /*dest_offset*/) {
     return -1;
}

extern int vx_new_start(vx_device_h /*hdevice*/, cmdbuffer* /*cmdBuf*/) {
    return -1;
}

extern int vx_start(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_ready_wait(vx_device_h /*hdevice*/, uint64_t /*timeout*/) {
    return -1;
}