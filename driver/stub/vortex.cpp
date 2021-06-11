#include <vortex.h>

extern int vx_dev_open(vx_device_h* /*hdevice*/) {
    return -1;
}

extern int vx_dev_close(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_dev_caps(vx_device_h /*hdevice*/, unsigned /*caps_id*/, unsigned* /*value*/) {
    return -1;
}

extern int vx_alloc_dev_mem(vx_device_h /*hdevice*/, size_t /*size*/, size_t* /*dev_maddr*/) {
    return -1;
}

extern int vx_alloc_shared_mem(vx_device_h /*hdevice*/, size_t /*size*/, vx_buffer_h* /*hbuffer*/) {
    return -1;
}

extern void* vx_host_ptr(vx_buffer_h /*hbuffer*/) {
    return nullptr;
}

extern int vx_buf_release(vx_buffer_h /*hbuffer*/) {
    return -1;
}

extern int vx_copy_to_dev(vx_buffer_h /*hbuffer*/, size_t /*dev_maddr*/, size_t /*size*/, size_t /*src_offset*/) {
    return -1;
}

extern int vx_copy_from_dev(vx_buffer_h /*hbuffer*/, size_t /*dev_maddr*/, size_t /*size*/, size_t /*dest_offset*/) {
     return -1;
}

extern int vx_start(vx_device_h /*hdevice*/) {
    return -1;
}

extern int vx_ready_wait(vx_device_h /*hdevice*/, long long /*timeout*/) {
    return -1;
}