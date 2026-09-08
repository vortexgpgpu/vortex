// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

#include "dxa/dxa_core.h"
#include "core.h"
#include "processor.h"
#include "mem_block_pool.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

using namespace vortex;

int main() {
  Processor processor;
  auto dxa = DxaCore::Create("s2g-worker-test", nullptr);
  SimPlatform::instance().reset();
  constexpr uint32_t bytes = 48 * VX_CFG_L1_LINE_SIZE + 19;
  constexpr uint32_t source_base = 517, destination_base = 8199;
  std::vector<uint8_t> source(8192), output(16384, 0x6d), expected = output;
  for (uint32_t i = 0; i < source.size(); ++i) {
    source[i] = uint8_t(i * 71 + i / 37);
  }
  std::memcpy(expected.data() + destination_base, source.data() + source_base, bytes);
  const uint32_t dcr = VX_DCR_DXA_DESC_BASE;
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, destination_base);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SIZE0_OFF, bytes);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_META_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, bytes);
  DxaReq operation{};
  operation.core = processor.get_first_core();
  operation.uuid = 17;
  operation.wid = 0;
  operation.direction = DxaDirection::S2G;
  operation.group_id = 3;
  operation.cta_mask = 1;
  operation.smem_addr = source_base;
  dxa->dxa_req_in[0].send(operation);

  struct Pending { MemReq request; uint32_t accepted; };
  std::vector<Pending> pending;
  std::mt19937 random(5371);
  uint32_t reads = 0, stores = 0, events = 0, peak = 0, reordered = 0;
  for (uint32_t cycle = 0; cycle < 30000; ++cycle) {
    SimPlatform::instance().tick();
    auto& reads_out = dxa->lmem_req_out[0];
    if (!reads_out.empty() && random() % 3 != 0) {
      if (events) {
        throw std::logic_error("source request after SOURCE_CONSUMED");
      }
      pending.push_back({reads_out.peek(), cycle});
      reads_out.pop();
      ++reads;
      peak = std::max<uint32_t>(peak, pending.size());
    }
    if (!pending.empty() && random() % 3 == 0) {
      uint32_t index = random() % pending.size();
      const auto& p = pending[index];
      if (cycle - p.accepted >= 9) {
        auto payload = make_mem_block();
        std::memcpy(payload->data(), source.data() + (p.request.addr & ~(uint64_t(VX_CFG_MEM_BLOCK_SIZE) - 1)), payload->size());
        MemRsp response(p.request.tag, 0, p.request.uuid, payload);
        if (dxa->lmem_rsp_in[0].try_send(response)) {
          reordered += index != 0;
          pending.erase(pending.begin() + index);
        }
      }
    }
    auto& completion = dxa->completion_out[0];
    if (!completion.empty()) {
      const auto& token = completion.peek();
      if (token.wid != operation.wid || token.group_id != operation.group_id || ++events != 1 || !pending.empty()) {
        throw std::logic_error("invalid architectural source completion");
      }
      completion.pop();
      std::fill(source.begin(), source.end(), 0xa5);
    }
    auto& writes_out = dxa->gmem_req_out[0];
    if (!writes_out.empty() && random() % 5 == 0) {
      const auto& store = writes_out.peek();
      if (!store.is_write() || !store.data) {
        throw std::logic_error("S2G emitted a non-store request");
      }
      for (uint32_t i = 0; i < VX_CFG_MEM_BLOCK_SIZE; ++i) {
        if (store.byteen & (uint64_t(1) << i)) {
          output.at(store.addr + i) = store.data->at(i);
        }
      }
      writes_out.pop();
      ++stores;
    }
    if (events && !dxa->running()) {
      if (output != expected || events != 1) {
        throw std::logic_error("S2G data/canary mismatch");
      }
#if defined(VX_CFG_DXA_S2G_PIPELINED) && defined(VX_CFG_DXA_S2G_PIPE_MULTI_READ)
      if (VX_CFG_DXA_S2G_READ_CREDITS > 1 && (peak < 2 || !reordered)) {
        throw std::logic_error("S2G pipeline did not exercise outstanding/reordered reads");
      }
#endif
      std::cout << "S2G worker: cycles=" << cycle << " reads=" << reads << " stores=" << stores
                << " source_events=" << events << " peak=" << peak << " reordered=" << reordered << " PASSED\n";
      return 0;
    }
  }
  throw std::logic_error("S2G worker timed out");
}
