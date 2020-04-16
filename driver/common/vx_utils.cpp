#include <iostream>
#include <fstream>
#include <cstring>
#include <vortex.h>
#include <config.h>

extern int vx_dev_caps(int caps_id) {
  switch (caps_id) {
  case VX_CAPS_VERSION:
    return 0;
  case VX_CAPS_MAX_CORES:
    return NUM_CORES;
  case VX_CAPS_MAX_WARPS:
    return NUM_WARPS;
  case VX_CAPS_MAX_THREADS:
    return NUM_THREADS;
  case VX_CAPS_CACHE_LINESIZE:
    return GLOBAL_BLOCK_SIZE_BYTES;
  case VX_CAPS_LOCAL_MEM_SIZE:
    return 0xffffffff;
  case VX_CAPS_ALLOC_BASE_ADDR:
    return 0x10000000;
  case VX_CAPS_KERNEL_BASE_ADDR:
    return 0x80000000;
  default:
    std::cout << "invalid caps id: " << caps_id << std::endl;
    std::abort();
    return 0;
  }
}

extern int vx_upload_kernel_bytes(vx_device_h device, const void* content, size_t size) {
  int err = 0;

  if (NULL == content || 0 == size)
    return -1;

  uint32_t buffer_transfer_size = 65536;
  uint32_t kernel_base_addr = vx_dev_caps(VX_CAPS_KERNEL_BASE_ADDR);

  // allocate device buffer
  vx_buffer_h buffer;
  err = vx_alloc_shared_mem(device, buffer_transfer_size, &buffer);
  if (err != 0)
    return -1; 

  // get buffer address
  auto buf_ptr = (uint8_t*)vx_host_ptr(buffer);

 #if defined(USE_SIMX)
  // default startup routine
  ((uint32_t*)buf_ptr)[0] = 0xf1401073;
  ((uint32_t*)buf_ptr)[1] = 0xf1401073;      
  ((uint32_t*)buf_ptr)[2] = 0x30101073;
  ((uint32_t*)buf_ptr)[3] = 0x800000b7;
  ((uint32_t*)buf_ptr)[4] = 0x000080e7;
  err = vx_copy_to_dev(buffer, 0, 5 * 4, 0);
  if (err != 0) {
    vx_buf_release(buffer);
    return err;
  }

  // newlib io simulator trap
  ((uint32_t*)buf_ptr)[0] = 0x00008067;
  err = vx_copy_to_dev(buffer, 0x70000000, 4, 0);
  if (err != 0) {
    vx_buf_release(buffer);
    return err;
  }
#endif

  //
  // upload content
  //

  size_t offset = 0;
  while (offset < size) {
    auto chunk_size = std::min<size_t>(buffer_transfer_size, size - offset);
    std::memcpy(buf_ptr, (uint8_t*)content + offset, chunk_size);
    err = vx_copy_to_dev(buffer, kernel_base_addr + offset, chunk_size, 0);
    if (err != 0) {
      vx_buf_release(buffer);
      return err;
    }
    offset += chunk_size;
  }

  vx_buf_release(buffer);

  return 0;
}

extern int vx_upload_kernel_file(vx_device_h device, const char* filename) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "error: " << filename << " not found" << std::endl;
    return -1;
  }

  // get length of file:
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  // allocate buffer 
  auto content = new char [size];  

  // read file content
  ifs.read(content, size);

  // upload
  int err = vx_upload_kernel_bytes(device, content, size);

  // release buffer
  delete[] content;

  return err;
}