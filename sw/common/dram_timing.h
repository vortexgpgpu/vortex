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

// The DRAM devices the platform memory models (VX_CFG_PLATFORM_MEMORY_TYPE):
// each JEDEC standard's organization and timing as a data sheet gives it, and
// the conversion of its nanosecond timings to DRAM clocks at a data rate.
// sim/common/dram_sim.cpp hands these to Ramulator; the bank model
// (sim/common/dram_bank_model.h) and the Foldsim runtime convert them with the
// same rounding Ramulator applies, so all of them see the same clocks.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vortex {
namespace dram_timing {

enum class Standard {
  HBM2,   // HBM2/HBM2E: 128-bit channels of two 64-bit pseudo-channels, 32 B bursts
  HBM3,   // HBM3, a 128-bit channel per pair of 64-bit channels, 32 B bursts
  GDDR6,  // GDDR6 x16 channels, 32 B bursts, 2 KB pages
};

inline Standard parse_standard(const std::string& name) {
  if (name == "HBM2") return Standard::HBM2;
  if (name == "HBM3") return Standard::HBM3;
  if (name == "GDDR6") return Standard::GDDR6;
  throw std::invalid_argument("unknown DRAM standard " + name);
}

// Bus timings are DRAM clocks, fixed by the burst; array timings are
// nanoseconds, converted at the configured data rate. Names are Ramulator's.
struct clock_timing_t {
  const char* name;
  int clocks;
};

struct ns_timing_t {
  const char* name;
  float ns;
};

// A 32 B burst is BL4 on a 64-bit pseudo-channel: two clocks, and the same
// spacing between column commands to another bank group (four within one).
// nRTW only completes the set Ramulator requires.
inline constexpr clock_timing_t hbm_bus_timing[] = {
  {"nBL", 2}, {"nCCDS", 2}, {"nCCDL", 4}, {"nRTW", 3},
};

// JEDEC HBM2 array timing as GPGPU-Sim's SM7_QV100 configuration lists it (its
// JEDEC line, at 1 GHz: RCD=RP=CL=14, RAS=33, RC=47, WR=12, RRD=4, RTPL=4);
// the rest from Ramulator's HBM2_2Gbps preset, whose 1 ns clock makes its
// cycles nanoseconds. The array does not speed up with the interface: HBM2E
// and HBM3 parts keep these timings (GPGPU-Sim's SM80_A100 and SM90_H100 HBM
// lines are the same 14/33/47 ns at their own clocks). Refresh timings complete
// the set Ramulator requires.
inline constexpr ns_timing_t hbm_array_timing[] = {
  {"tCL", 14}, {"tRCDRD", 14}, {"tRCDWR", 14}, {"tRP", 14}, {"tRAS", 33}, {"tRC", 47},
  {"tCWL", 2}, {"tWR", 12}, {"tRTPS", 2}, {"tRTPL", 4}, {"tRRDS", 4}, {"tRRDL", 4},
  {"tWTRS", 3}, {"tWTRL", 4}, {"tFAW", 15}, {"tRFCSB", 160}, {"tREFI", 3900}, {"tRREFD", 8},
};

// Ramulator's GDDR6 clock is half the data rate: a 32 B burst, 16 beats on a
// x16 channel, is eight clocks, and column commands are two (three within a
// bank group) command clocks apart, a command clock being eight data beats.
inline constexpr clock_timing_t gddr6_bus_timing[] = {
  {"nBL", 8}, {"nCCDS", 8}, {"nCCDL", 12},
};

// Ramulator's x16 GDDR6 timing at 2000 (its GDDR6_2000_1350mV_double preset
// and secondary tables) at its 570 ps command clock, in nanoseconds; refresh
// timings complete the set Ramulator requires.
inline constexpr ns_timing_t gddr6_array_timing[] = {
  {"tCL", 13.7f}, {"tRCDRD", 14.8f}, {"tRCDWD", 9.1f}, {"tRP", 14.8f}, {"tRAS", 30.2f}, {"tRC", 45.0f},
  {"tWR", 14.8f}, {"tRTP", 2.3f}, {"tCWL", 3.4f}, {"tRRDS", 2.9f}, {"tRRDL", 3.4f},
  {"tWTRS", 5.1f}, {"tWTRL", 6.3f}, {"tFAW", 16.0f},
  {"tRFC", 120.0f}, {"tRFCpb", 60.0f}, {"tREFI", 1900.0f}, {"tRREFD", 8.0f},
};

// One channel's organization and the address map the simulators lay over it.
struct geometry_t {
  uint32_t log2_pcs;          // pseudo-channels sharing the channel's commands
  uint32_t log2_bgs;          // bank groups per pseudo-channel
  uint32_t log2_banks;        // banks per bank group
  uint32_t log2_row_bursts;   // 32 B bursts per row
  uint32_t log2_visit;        // bursts per row visit before the next bank
  bool     two_clock_act;     // an ACT holds the row-command bus for two clocks
  uint32_t rtw_extra;         // read-to-write turnaround beyond tCL + tBL - tCWL
  uint32_t bank_bits() const { return log2_pcs + log2_bgs + log2_banks; }
};

// HBM: 2 pseudo-channels x 4 bank groups x 4 banks, 1 KB rows (32 bursts);
// GDDR6 x16: 4 x 4 banks, 2 KB rows (64 bursts). Either way a row visit is four
// bursts (minimalist open page, 128 B). Read-to-write is Ramulator's: HBM's
// assumes a one-clock write preamble, GDDR6's adds the bus turnaround.
inline geometry_t geometry(Standard standard) {
  if (standard == Standard::GDDR6) {
    return geometry_t{0, 2, 2, 6, 2, false, 4};
  }
  return geometry_t{1, 2, 2, 5, 2, true, 2};
}

// Every device here holds 8 Gb per channel; addresses above alias.
inline constexpr uint64_t kChannelSize = uint64_t(1) << 30;
// Each standard moves 32 B per column command (prefetch x channel width).
inline constexpr uint32_t kBurstSize = 32;

// Ramulator's DRAM clock period at a data rate (MT/s): half the data rate, in
// integer picoseconds.
inline int tck_ps(uint32_t data_rate) {
  return int(1E6 / (data_rate / 2));
}

// Ramulator's JEDEC rounding of a nanosecond timing to clocks (base/utils.cpp),
// in the same float arithmetic.
inline uint32_t jedec_clocks(float t_ns, int tck) {
  uint64_t t_ps = t_ns * 1000;
  return uint32_t(((t_ps * 1000 / tck) + 974) / 1000);
}

// The timings a scheduler of one channel needs, in DRAM clocks.
struct clocks_t {
  uint32_t BL, CCDS, CCDL;
  uint32_t CL, CWL;
  uint32_t RCDRD, RCDWR, RP, RAS, RC;
  uint32_t RTP, WR;
  uint32_t RRDS, RRDL, FAW;
  uint32_t WTRS, WTRL;
};

namespace detail {

template <size_t N>
inline bool find_clock(const clock_timing_t (&table)[N], const char* name, uint32_t* out) {
  for (auto& t : table) {
    if (std::strcmp(t.name, name) == 0) {
      *out = uint32_t(t.clocks);
      return true;
    }
  }
  return false;
}

template <size_t N>
inline bool find_ns(const ns_timing_t (&table)[N], const char* name, int tck, uint32_t* out) {
  for (auto& t : table) {
    if (std::strcmp(t.name, name) == 0) {
      *out = jedec_clocks(t.ns, tck);
      return true;
    }
  }
  return false;
}

template <size_t B, size_t A>
inline uint32_t get(const clock_timing_t (&bus)[B], const ns_timing_t (&array)[A], int tck,
                    const char* clock_name, const char* ns_name) {
  uint32_t v = 0;
  if (clock_name && find_clock(bus, clock_name, &v)) return v;
  if (ns_name && find_ns(array, ns_name, tck, &v)) return v;
  throw std::logic_error(std::string("dram_timing: no timing ") + (ns_name ? ns_name : clock_name));
}

} // namespace detail

// The standard's timings at a data rate. A read may precharge its bank tRTP
// after its column command: HBM's bank-level read-to-precharge is the larger of
// its short and long forms.
inline clocks_t clocks(Standard standard, uint32_t data_rate) {
  const int tck = tck_ps(data_rate);
  clocks_t c{};
  if (standard == Standard::GDDR6) {
    auto g = [&](const char* n, const char* t) {
      return detail::get(gddr6_bus_timing, gddr6_array_timing, tck, n, t);
    };
    c.BL = g("nBL", nullptr); c.CCDS = g("nCCDS", nullptr); c.CCDL = g("nCCDL", nullptr);
    c.CL = g(nullptr, "tCL"); c.CWL = g(nullptr, "tCWL");
    c.RCDRD = g(nullptr, "tRCDRD"); c.RCDWR = g(nullptr, "tRCDWD");
    c.RP = g(nullptr, "tRP"); c.RAS = g(nullptr, "tRAS"); c.RC = g(nullptr, "tRC");
    c.RTP = g(nullptr, "tRTP"); c.WR = g(nullptr, "tWR");
    c.RRDS = g(nullptr, "tRRDS"); c.RRDL = g(nullptr, "tRRDL"); c.FAW = g(nullptr, "tFAW");
    c.WTRS = g(nullptr, "tWTRS"); c.WTRL = g(nullptr, "tWTRL");
  } else {
    auto g = [&](const char* n, const char* t) {
      return detail::get(hbm_bus_timing, hbm_array_timing, tck, n, t);
    };
    c.BL = g("nBL", nullptr); c.CCDS = g("nCCDS", nullptr); c.CCDL = g("nCCDL", nullptr);
    c.CL = g(nullptr, "tCL"); c.CWL = g(nullptr, "tCWL");
    c.RCDRD = g(nullptr, "tRCDRD"); c.RCDWR = g(nullptr, "tRCDWR");
    c.RP = g(nullptr, "tRP"); c.RAS = g(nullptr, "tRAS"); c.RC = g(nullptr, "tRC");
    uint32_t rtps = g(nullptr, "tRTPS"), rtpl = g(nullptr, "tRTPL");
    c.RTP = (rtps > rtpl) ? rtps : rtpl;
    c.WR = g(nullptr, "tWR");
    c.RRDS = g(nullptr, "tRRDS"); c.RRDL = g(nullptr, "tRRDL"); c.FAW = g(nullptr, "tFAW");
    c.WTRS = g(nullptr, "tWTRS"); c.WTRL = g(nullptr, "tWTRL");
  }
  return c;
}

} // namespace dram_timing
} // namespace vortex
