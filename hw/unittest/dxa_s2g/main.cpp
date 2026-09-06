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

#include "vl_simulator.h"
#include "VVX_dxa_s2g_top.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using Device = VVX_dxa_s2g_top;

static constexpr uint32_t kLineBytes = 64;
static constexpr uint32_t kLineWords = kLineBytes / sizeof(uint32_t);
static constexpr uint32_t kLmemWordBytes = 16;
static constexpr uint32_t kLmemWords = kLmemWordBytes / sizeof(uint32_t);
static constexpr uint32_t kMemoryBytes = 65536;

static uint64_t timestamp = 0;
static int checks_total = 0;
static int checks_failed = 0;

double sc_time_stamp() { return timestamp; }

#define CHECKX(cond, ...)                                      \
  do {                                                         \
    ++checks_total;                                            \
    if (!(cond)) {                                             \
      std::printf("  FAILED (t=%llu): %s -- ",                \
                  (unsigned long long)timestamp, #cond);       \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                      \
      ++checks_failed;                                         \
    }                                                          \
  } while (0)

struct Transfer {
  const char *name;
  uint32_t gmem_base;
  uint32_t smem_base;
  uint32_t row_bytes;
  uint32_t rows;
  uint32_t valid_rows;
  uint32_t row_stride;
  uint32_t expected_stores;
  bool gate_issue;
};

struct Bench {
  vl_simulator<Device> sim;
  std::array<uint8_t, kMemoryBytes> lmem{};
  std::array<uint8_t, kMemoryBytes> gmem{};
  std::array<uint8_t, kMemoryBytes> initial_gmem{};
  bool lmem_pending = false;
  uint32_t lmem_delay = 0;
  uint32_t lmem_tag = 0;
  std::array<uint8_t, kLmemWordBytes> lmem_rsp{};
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
  struct PendingRead {
    uint32_t tag;
    uint32_t delay;
    std::array<uint8_t, kLmemWordBytes> data;
  };
  std::vector<PendingRead> lmem_queue;
  int response_index = -1;
  size_t max_lmem_pending = 0;
#endif
  uint32_t lmem_reads = 0;
  uint32_t stores = 0;
  uint32_t requests = 0;
  std::vector<uint64_t> events;
  bool completion_held = false;
  uint32_t held_event = 0;

  Bench() {
    for (uint32_t i = 0; i < kMemoryBytes; ++i) {
      lmem[i] = uint8_t((i * 37u + 11u) & 0xffu);
      initial_gmem[i] = uint8_t((i * 13u + 0xa5u) & 0xffu);
    }
  }

  static void set_lmem_rsp(VlWide<kLmemWords> &dst,
                           const std::array<uint8_t, kLmemWordBytes> &src) {
    for (uint32_t word = 0; word < kLmemWords; ++word) {
      uint32_t value = 0;
      for (uint32_t byte = 0; byte < 4; ++byte)
        value |= uint32_t(src[word * 4 + byte]) << (byte * 8);
      dst[word] = value;
    }
  }

  static uint8_t wide_byte(const VlWide<kLineWords> &src, uint32_t index) {
    return uint8_t((src[index / 4] >> ((index % 4) * 8)) & 0xffu);
  }

  void reset() {
    sim->issue_allowed = 1;
    sim->req_valid = 0;
    sim->lmem_req_ready = 0;
    sim->lmem_rsp_valid = 0;
    sim->gmem_req_ready = 0;
    sim->gmem_rsp_valid = 0;
    sim->completion_ready = 0;
    lmem_pending = false;
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
    lmem_queue.clear();
    response_index = -1;
    max_lmem_pending = 0;
#endif
    lmem_delay = 0;
    lmem_reads = 0;
    stores = 0;
    requests = 0;
    events.clear();
    completion_held = false;
    gmem = initial_gmem;
    timestamp = sim.reset(timestamp);
  }

  void drive_inputs(bool completion_ready) {
    sim->lmem_req_ready = ((timestamp / 2) % 4) != 0;
    sim->gmem_req_ready = ((timestamp / 2) % 5) != 1;
    sim->completion_ready = completion_ready;
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
    response_index = -1;
    for (int i = int(lmem_queue.size()) - 1; i >= 0; --i) {
      if (lmem_queue[i].delay == 0) {
        response_index = i;
        break;
      }
    }
    sim->lmem_rsp_valid = response_index >= 0;
    if (response_index >= 0) {
      sim->lmem_rsp_tag = lmem_queue[response_index].tag;
      set_lmem_rsp(sim->lmem_rsp_data, lmem_queue[response_index].data);
    } else {
      sim->lmem_rsp_tag = 0;
      set_lmem_rsp(sim->lmem_rsp_data, lmem_rsp);
    }
#else
    sim->lmem_rsp_valid = lmem_pending && (lmem_delay == 0);
    sim->lmem_rsp_tag = lmem_tag;
    set_lmem_rsp(sim->lmem_rsp_data, lmem_rsp);
#endif
    sim->gmem_rsp_valid = 0;
    sim->gmem_rsp_tag = 0;
  }

  void tick(bool completion_ready = true) {
    drive_inputs(completion_ready);
    sim->eval();
    const uint64_t cycle = timestamp / 2;

    if (sim->req_valid && sim->req_ready)
      ++requests;

    if (sim->lmem_req_valid && sim->lmem_req_ready) {
      CHECKX(!sim->lmem_req_rw, "S2G issued an LMEM write");
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
      PendingRead pending;
      pending.tag = sim->lmem_req_tag;
      pending.delay = 4 + (lmem_reads % 4);
      const uint32_t base = sim->lmem_req_addr * kLmemWordBytes;
      CHECKX(base + kLmemWordBytes <= lmem.size(), "LMEM request out of range");
      for (uint32_t i = 0; i < kLmemWordBytes; ++i)
        pending.data[i] = lmem[base + i];
      lmem_queue.push_back(pending);
      max_lmem_pending = std::max(max_lmem_pending, lmem_queue.size());
#else
      CHECKX(!lmem_pending, "more than one LMEM read is outstanding");
      lmem_pending = true;
      lmem_delay = 1 + (lmem_reads % 4);
      lmem_tag = sim->lmem_req_tag;
      const uint32_t base = sim->lmem_req_addr * kLmemWordBytes;
      CHECKX(base + kLmemWordBytes <= lmem.size(), "LMEM request out of range");
      for (uint32_t i = 0; i < kLmemWordBytes; ++i)
        lmem_rsp[i] = lmem[base + i];
#endif
      ++lmem_reads;
    }

    if (sim->lmem_rsp_valid && sim->lmem_rsp_ready) {
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
      lmem_queue.erase(lmem_queue.begin() + response_index);
#else
      lmem_pending = false;
#endif
    }

    if (sim->gmem_req_valid && sim->gmem_req_ready) {
      CHECKX(sim->gmem_req_rw, "S2G issued a global-memory read");
      const uint32_t base = sim->gmem_req_addr * kLineBytes;
      CHECKX(base + kLineBytes <= gmem.size(), "global store out of range");
      for (uint32_t i = 0; i < kLineBytes; ++i) {
        if ((sim->gmem_req_byteen >> i) & 1ull)
          gmem[base + i] = wide_byte(sim->gmem_req_data, i);
      }
      ++stores;
    }

    if (completion_held) {
      CHECKX(sim->completion_valid, "completion valid dropped under stall");
      const uint32_t event = (uint32_t(sim->completion_core_id) << 16)
                           | (uint32_t(sim->completion_wid) << 12)
                           | (uint32_t(sim->completion_epoch) << 10)
                           | (uint32_t(sim->completion_group_seq) << 8)
                           | uint32_t(sim->completion_op_id);
      CHECKX(event == held_event, "completion payload changed under stall");
    }

    if (sim->completion_valid && !sim->completion_ready) {
      completion_held = true;
      held_event = (uint32_t(sim->completion_core_id) << 16)
                 | (uint32_t(sim->completion_wid) << 12)
                 | (uint32_t(sim->completion_epoch) << 10)
                 | (uint32_t(sim->completion_group_seq) << 8)
                 | uint32_t(sim->completion_op_id);
    } else {
      completion_held = false;
    }

    if (sim->completion_valid && sim->completion_ready) {
      CHECKX(sim->completion_core_id == 0, "completion core mismatch");
      CHECKX(sim->completion_wid == 1, "completion warp mismatch");
      CHECKX(sim->completion_epoch == 2, "completion epoch mismatch");
      CHECKX(sim->completion_group_seq == 0x5a, "completion sequence mismatch");
      CHECKX(sim->completion_op_id == 3, "completion op-id mismatch");
      events.push_back(cycle);
    }

    timestamp = sim.step(timestamp, 2);
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
    for (auto &pending : lmem_queue) {
      if (pending.delay != 0)
        --pending.delay;
    }
#else
    if (lmem_pending && lmem_delay != 0)
      --lmem_delay;
#endif
  }

  void drive_transfer(const Transfer &transfer) {
    sim->req_core_id = 0;
    sim->req_uuid = 1;
    sim->req_wid = 1;
    sim->req_epoch = 2;
    sim->req_group_seq = 0x5a;
    sim->req_op_id = 3;
    sim->req_smem_addr = transfer.smem_base;
    sim->req_meta = 0;
    sim->req_coords[0] = 0;
    sim->req_coords[1] = 0;
    sim->req_coords[2] = 0;
    sim->req_coords[3] = 0;
    sim->req_coords[4] = 0;
    sim->req_cta_mask = 1;
    sim->desc_base_addr = transfer.gmem_base;
    sim->desc_meta = (transfer.rows == 1 && transfer.valid_rows == 1) ? 1 : 2;
    sim->desc_tile01 = (transfer.rows << 16) | transfer.row_bytes;
    sim->desc_tile23 = 0;
    sim->desc_tile4 = 0;
    sim->desc_size0 = transfer.row_bytes;
    sim->desc_size1 = transfer.valid_rows;
    sim->desc_size2 = 1;
    sim->desc_size3 = 1;
    sim->desc_size4 = 1;
    sim->desc_stride0 = transfer.row_stride;
    sim->desc_stride1 = 0;
    sim->desc_stride2 = 0;
    sim->desc_stride3 = 0;

    sim->issue_allowed = !transfer.gate_issue;
    sim->req_valid = 1;
    for (uint32_t cycle = 0; cycle < 10 && transfer.gate_issue; ++cycle) {
      tick();
      CHECKX(!sim->req_ready, "request bypassed the issue gate");
    }
    sim->issue_allowed = 1;
    sim->eval();
    for (uint32_t cycle = 0; cycle < 100 && !sim->req_ready; ++cycle)
      tick();
    CHECKX(sim->req_ready, "S2G request was not accepted");
    tick();
    sim->req_valid = 0;
  }

  void verify_memory(const Transfer &transfer) {
    std::array<uint8_t, kMemoryBytes> expected = initial_gmem;
    for (uint32_t row = 0; row < transfer.valid_rows; ++row) {
      const uint32_t dst = transfer.gmem_base + row * transfer.row_stride;
      const uint32_t src = transfer.smem_base + row * transfer.row_bytes;
      for (uint32_t byte = 0; byte < transfer.row_bytes; ++byte)
        expected[dst + byte] = lmem[src + byte];
    }
    for (uint32_t i = 0; i < gmem.size(); ++i)
      CHECKX(gmem[i] == expected[i], "global byte %u got=0x%02x want=0x%02x",
             i, gmem[i], expected[i]);
  }

  void run(const Transfer &transfer) {
    std::printf("[%s]\n", transfer.name);
    reset();
    drive_transfer(transfer);

    uint32_t read_stall = 0;
    for (uint32_t cycle = 0; cycle < 4000 && events.empty(); ++cycle) {
      bool ready = true;
      if (sim->completion_valid && read_stall < 9) {
        ready = false;
        ++read_stall;
      }
      tick(ready);
    }
    for (uint32_t cycle = 0; cycle < 100 && sim->busy; ++cycle)
      tick();

    CHECKX(events.size() == 1, "completion count got=%zu want=1", events.size());
    CHECKX(requests == 1, "request handshakes got=%u want=1", requests);
#ifdef VX_CFG_DXA_S2G_PIPE_MULTI_READ
    if (transfer.expected_stores > 1)
      CHECKX(max_lmem_pending > 1, "multi-read never exceeded one LMEM credit");
#endif
    CHECKX(stores == transfer.expected_stores, "store beats got=%u want=%u",
           stores, transfer.expected_stores);
    sim->eval();
    CHECKX(!sim->busy, "worker remained busy after READ completion and last store issue");
    verify_memory(transfer);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  static_assert(kLineBytes == 64, "unit test expects the default cache line");
  static_assert(kLmemWordBytes == 16, "unit test expects the default LMEM DMA word");

  Bench bench;
  bench.run(Transfer{"unaligned two-line copy", 0x1003, 37, 100, 1, 1, 0, 2, true});
  bench.run(Transfer{"single partial with LMEM crossing", 0x200d, 255, 29, 1, 1, 0, 1, false});
  bench.run(Transfer{"two valid rows plus OOB row", 0x3005, 511, 20, 3, 2, 64, 2, false});
  bench.run(Transfer{"four-line credit window", 0x5000, 1024, 200, 1, 1, 0, 4, false});
  bench.run(Transfer{"fully OOB transfer", 0x4007, 777, 23, 1, 0, 64, 0, false});

  std::printf("checks=%d failed=%d\n", checks_total, checks_failed);
  if (checks_failed != 0) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
