#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
#include <assert.h>
#include "fpga.h"
#include "opae_sim.h"
#include <VX_config.h>

using namespace vortex;

extern fpga_result fpgaOpen(fpga_token token, fpga_handle *handle, int flags) {
  if (NULL == handle || flags != 0)
    return FPGA_INVALID_PARAM;
  auto sim = new opae_sim();    
  *handle = reinterpret_cast<fpga_handle>(sim);
  return FPGA_OK;
}

extern fpga_result fpgaClose(fpga_handle handle) {
  if (NULL == handle)
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  delete sim;
  
  return FPGA_OK;
}

extern fpga_result fpgaPrepareBuffer(fpga_handle handle, uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
  if (NULL == handle || len == 0 || buf_addr == NULL || wsid == NULL) 
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  int ret = sim->prepare_buffer(len, buf_addr, wsid, flags);
  if (ret != 0)
    return FPGA_NO_MEMORY;
  
  return FPGA_OK;
}

extern fpga_result fpgaReleaseBuffer(fpga_handle handle, uint64_t wsid) {
  if (NULL == handle) 
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  sim->release_buffer(wsid);

  return FPGA_OK;
}

extern fpga_result fpgaGetIOAddress(fpga_handle handle, uint64_t wsid, uint64_t *ioaddr) {
  if (NULL == handle || ioaddr == NULL) 
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  sim->get_io_address(wsid, ioaddr);

  return FPGA_OK;
}

extern fpga_result fpgaWriteMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t value) {
  if (NULL == handle || mmio_num != 0) 
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  sim->write_mmio64(mmio_num, offset, value);

  return FPGA_OK;
}

extern fpga_result fpgaReadMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t *value) {
  if (NULL == handle || mmio_num != 0 || value == NULL) 
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  sim->read_mmio64(mmio_num, offset, value);

  return FPGA_OK;
}

extern const char *fpgaErrStr(fpga_result e) {
  return "";
}