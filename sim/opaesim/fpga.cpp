// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#include <util.h>

using namespace vortex;

#ifdef __cplusplus
extern "C" {
#endif

extern fpga_result fpgaGetProperties(fpga_token token, fpga_properties *prop) {
  __unused (token, prop);
  return FPGA_OK;
}

extern fpga_result fpgaPropertiesSetObjectType(fpga_properties prop, fpga_objtype objtype) {
  __unused (prop, objtype);
  return FPGA_OK;
}

extern fpga_result fpgaPropertiesSetGUID(fpga_properties prop, fpga_guid guid) {
  __unused (prop, guid);
  return FPGA_OK;
}

extern fpga_result fpgaDestroyProperties(fpga_properties *prop) {
  __unused (prop);
  return FPGA_OK;
}

extern fpga_result fpgaEnumerate(const fpga_properties *filters, uint32_t num_filters, fpga_token *tokens, uint32_t max_tokens, uint32_t *num_matches) {
  __unused (filters, num_filters, num_filters, tokens, max_tokens);
  if (num_matches) {
    *num_matches = 1;
  }
  return FPGA_OK;
}

extern fpga_result fpgaDestroyToken(fpga_token *token) {
  __unused (token);
  return FPGA_OK;
}

extern fpga_result fpgaPropertiesGetLocalMemorySize(const fpga_properties *filters, uint64_t* lms) {
  __unused (filters);
  if (lms) {
  #if (XLEN == 64)
    *lms = 0x200000000; // 8 GB
  #else
    *lms = 0x100000000; // 4 GB
  #endif
  }
  return FPGA_OK;
}

extern fpga_result fpgaOpen(fpga_token token, fpga_handle *handle, int flags) {
  __unused (token);
  if (NULL == handle || flags != 0)
    return FPGA_INVALID_PARAM;
  auto sim = new opae_sim();
  int ret = sim->init();
  if (ret != 0) {
    delete sim;
    return FPGA_NO_MEMORY;
  }
  *handle = reinterpret_cast<fpga_handle>(sim);
  return FPGA_OK;
}

extern fpga_result fpgaClose(fpga_handle handle) {
  if (NULL == handle)
    return FPGA_INVALID_PARAM;

  auto sim = reinterpret_cast<opae_sim*>(handle);
  sim->shutdown();

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

#ifdef __cplusplus
}
#endif
