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
#include <deque>
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

// Stands in for the socket: feeds the DXA its request and GMEM responses.
class Driver : public SimObject<Driver> {
public:
  SimPortOut<DxaReq> req_out{this};
  SimPortOut<MemRsp> rsp_out{this};
  using SimObject::SimObject;
};

// Stands in for L2 and LMEM: records every DXA output for the test to inspect.
class Sink : public SimObject<Sink> {
public:
  std::vector<std::deque<MemReq>> gmem;
  std::vector<std::deque<MemReq>> lmem;
  SimPortInArray<MemReq> gmem_in;
  SimPortInArray<MemReq> lmem_in;

  Sink(const SimContext &ctx, const char *name, uint32_t gmem_ports,
       uint32_t lmem_ports)
      : SimObject<Sink>(ctx, name)
      , gmem(gmem_ports)
      , lmem(lmem_ports)
      , gmem_in(this, gmem_ports, 4, &Sink::take_gmem)
      , lmem_in(this, lmem_ports, 4, &Sink::take_lmem) {
  }

  bool take_gmem(uint32_t port, MemReq &req) {
    gmem.at(port).push_back(req);
    return true;
  }

  bool take_lmem(uint32_t port, MemReq &req) {
    lmem.at(port).push_back(req);
    return true;
  }

  void on_bar_release(const uint32_t &) {}
};

bool pop_front(std::deque<MemReq> &queue, MemReq *req) {
  if (queue.empty()) {
    return false;
  }
  *req = queue.front();
  queue.pop_front();
  return true;
}

MemReq wait_for_lmem_write(Sink *sink, uint32_t limit = 100) {
  for (uint32_t i = 0; i < limit; ++i) {
    tick();
    MemReq req;
    if (pop_front(sink->lmem.at(0), &req)) {
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

void send_response(Driver *driver, const MemReq &request, uint8_t value) {
  driver->rsp_out.send(MemRsp(request.tag, 0, 1, make_response_data(value)));
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
  auto driver = Driver::Create("driver");
  auto sink = Sink::Create("sink", uint32_t(dxa->gmem_req_out.size()),
                           uint32_t(dxa->lmem_req_out.size()));

  driver->req_out.bind(&dxa->dxa_req_in.at(0));
  driver->rsp_out.bind(dxa->gmem_rsp_in.at(0));
  for (uint32_t i = 0; i < dxa->gmem_req_out.size(); ++i) {
    dxa->gmem_req_out.at(i)->bind(&sink->gmem_in.at(i));
  }
  for (uint32_t i = 0; i < dxa->lmem_req_out.size(); ++i) {
    dxa->lmem_req_out.at(i).bind(&sink->lmem_in.at(i));
  }
  for (auto &link : dxa->bar_release_out) {
    link.bind(sink.get(), &Sink::on_bar_release);
  }

  SimPlatform::instance().reset();

  constexpr uint32_t slot = 0;
  constexpr uint32_t line_size = VX_CFG_L1_LINE_SIZE;
  constexpr uint32_t transfer_size = 3 * line_size;
  constexpr uint32_t gmem_base = 16 * line_size;
  constexpr uint32_t smem_base = 8 * line_size;
  constexpr uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, gmem_base);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SIZE0_OFF, transfer_size);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_META_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, transfer_size);

  DxaReq req{};
  req.core = core;
  req.uuid = 1;
  req.desc_slot = slot;
  req.cta_mask = 1;
  req.smem_addr = smem_base;
  driver->req_out.send(req);

  std::vector<MemReq> reads;
  for (uint32_t i = 0; i < 100 && reads.size() < 3; ++i) {
    tick();
    MemReq read;
    if (pop_front(sink->gmem.at(0), &read)) {
      reads.push_back(read);
    }
  }

  if (reads.size() != 3)
    fail("expected exactly three GMEM reads");
  if (reads.at(0).addr != gmem_base ||
      reads.at(1).addr != gmem_base + line_size ||
      reads.at(2).addr != gmem_base + 2 * line_size)
    fail("unexpected GMEM request addresses");

  // C is the last work item. Even if it returns first, its completion write
  // must remain deferred while A and B are outstanding.
  send_response(driver.get(), reads.at(2), 0xcc);
  tick(8);
  if (!sink->lmem.at(0).empty())
    fail("last response drained before older outstanding requests");

  // B is ready and is not the last work item, so it must bypass A.
  send_response(driver.get(), reads.at(1), 0xbb);
  check_write(wait_for_lmem_write(sink.get()), smem_base + line_size, 0xbb,
              false);

  // Once A completes, the previously deferred C response must drain last.
  send_response(driver.get(), reads.at(0), 0xaa);
  check_write(wait_for_lmem_write(sink.get()), smem_base, 0xaa, false);
  check_write(wait_for_lmem_write(sink.get()), smem_base + 2 * line_size, 0xcc,
              true);

  std::cout << "PASSED" << std::endl;
  return 0;
}
