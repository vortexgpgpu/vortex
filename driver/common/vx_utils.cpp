#include <iostream>
#include <fstream>
#include <cstring>
#include <vortex.h>
#include <VX_config.h>

extern int vx_upload_kernel_bytes(vx_device_h device, const void* content, size_t size) {
  int err = 0;

  if (NULL == content || 0 == size)
    return -1;

  uint32_t buffer_transfer_size = 65536;
  unsigned kernel_base_addr;
  err = vx_dev_caps(device, VX_CAPS_KERNEL_BASE_ADDR, &kernel_base_addr);
  if (err != 0)
    return -1;

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

  // read file content
  ifs.seekg(0, ifs.end);
  auto size = ifs.tellg();
  auto content = new char [size];   
  ifs.seekg(0, ifs.beg);
  ifs.read(content, size);

  // upload
  int err = vx_upload_kernel_bytes(device, content, size);

  // release buffer
  delete[] content;

  return err;
}

extern int vx_get_perf(vx_device_h device, int core_id, size_t* instrs, size_t* cycles) {
  int ret = 0;

  unsigned value;
  
  if (instrs) {
    ret |= vx_csr_get(device, core_id, CSR_INSTRET_H, &value);
    *instrs = value;
    ret |= vx_csr_get(device, core_id, CSR_INSTRET, &value);
    *instrs = (*instrs << 32) | value;
  }

  if (cycles) {
    ret |= vx_csr_get(device, core_id, CSR_CYCLE_H, &value);
    *cycles = value;
    ret |= vx_csr_get(device, core_id, CSR_CYCLE, &value);
    *cycles = (*cycles << 32) | value;
  }

  return ret;
}