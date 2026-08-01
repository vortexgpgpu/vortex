// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "core.h"
#include "dxa_core.h"
#include "mem_block_pool.h"
#include "processor.h"
#include "simobject.h"
#include <VX_types.h>

using namespace vortex;

namespace {

[[noreturn]] void fail(const char *message) {
  std::cerr << "FAILED: " << message << std::endl;
  std::exit(1);
}

void tick(uint32_t cycles = 1) {
  for (uint32_t i = 0; i < cycles; ++i)
    SimPlatform::instance().tick();
}

MemReq wait_for_lmem_write(DxaCore *dxa, uint32_t limit = 100) {
  for (uint32_t i = 0; i < limit; ++i) {
    tick();
    auto &output = dxa->lmem_req_out.at(0);
    if (!output.empty()) {
      MemReq req = output.peek();
      output.pop();
      return req;
    }
  }
  fail("timed out waiting for LMEM write");
}

std::shared_ptr<mem_block_t> make_response_data(uint8_t value) {
  auto data = make_mem_block();
  std::fill(data->begin(), data->end(), value);
  return data;
}

void send_response(DxaCore *dxa, const MemReq &request, uint8_t value) {
  dxa->gmem_rsp_in.at(0).send(
      MemRsp(request.tag, 0, 1, make_response_data(value)));
}

void check_write(const MemReq &write, uint64_t expected_addr,
                 uint8_t expected_data, bool expected_notify) {
  if (write.addr != expected_addr)
    fail("response wrote an unexpected LMEM address");
  if (!write.data || write.data->at(0) != expected_data)
    fail("response wrote unexpected data");
  if (write.flags.dxa_notify_done != expected_notify)
    fail("response carried an unexpected completion flag");
}

} // namespace

int main() {
  Processor processor;
  auto *core = processor.get_first_core();
  auto dxa = DxaCore::Create("dxa_ooo_test", nullptr);

  SimPlatform::instance().reset();

  constexpr uint32_t slot = 0;
  constexpr uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, 0x1000);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SIZE0_OFF, 192);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_META_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, 192);

  DxaReq req{};
  req.core = core;
  req.uuid = 1;
  req.desc_slot = slot;
  req.cta_mask = 1;
  req.smem_addr = 0x200;
  dxa->dxa_req_in.at(0).send(req);

  std::vector<MemReq> reads;
  for (uint32_t i = 0; i < 100 && reads.size() < 3; ++i) {
    tick();
    auto &output = dxa->gmem_req_out.at(0);
    if (!output.empty()) {
      reads.push_back(output.peek());
      output.pop();
    }
  }

  if (reads.size() != 3)
    fail("expected exactly three GMEM reads");
  if (reads.at(0).addr != 0x1000 || reads.at(1).addr != 0x1040 || reads.at(2).addr != 0x1080)
    fail("unexpected GMEM request addresses");

  // C is the last work item. Even if it returns first, its completion write
  // must remain deferred while A and B are outstanding.
  send_response(dxa.get(), reads.at(2), 0xcc);
  tick(8);
  if (!dxa->lmem_req_out.at(0).empty())
    fail("last response drained before older outstanding requests");

  // B is ready and is not the last work item, so it must bypass A.
  send_response(dxa.get(), reads.at(1), 0xbb);
  check_write(wait_for_lmem_write(dxa.get()), 0x240, 0xbb, false);

  // Once A completes, the previously deferred C response must drain last.
  send_response(dxa.get(), reads.at(0), 0xaa);
  check_write(wait_for_lmem_write(dxa.get()), 0x200, 0xaa, false);
  check_write(wait_for_lmem_write(dxa.get()), 0x280, 0xcc, true);

  std::cout << "PASSED" << std::endl;
  return 0;
}
