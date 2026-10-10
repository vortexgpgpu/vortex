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

#include "rtu_raylog.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_set>
#include "mem.h"
#include "types.h"

namespace vortex { namespace rtu { namespace raylog {

namespace {

template <typename T> uint32_t bits(T v) {
  static_assert(sizeof(T) == 4, "32-bit field");
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return u;
}

class Logger {
public:
  Logger() {
    const char* path = std::getenv("VX_RTU_RAYLOG");
    if (path == nullptr || path[0] == '\0') return;
    if (SIMX_NUM_WORKERS > 1) {
      // One log is shared by every RTU core; like debug tracing, it needs a
      // serial build.
      std::fprintf(stderr, "[rtu-raylog] disabled: requires a serial SimX build (SIMX_MT unset)\n");
      return;
    }
    fp_ = std::fopen(path, "wb");
    if (fp_ == nullptr) {
      std::fprintf(stderr, "[rtu-raylog] cannot open %s\n", path);
      return;
    }
    static char buf[1 << 20];
    std::setvbuf(fp_, buf, _IOFBF, sizeof(buf));
    if (const char* s = std::getenv("VX_RTU_RAYLOG_MAX"))   max_rays_ = std::strtoull(s, nullptr, 0);
    if (const char* s = std::getenv("VX_RTU_RAYLOG_EVERY")) every_    = std::max(1ull, std::strtoull(s, nullptr, 0));
    RaylogHeader h{};
    h.magic       = kMagic;
    h.version     = kVersion;
    h.num_threads = VX_CFG_NUM_THREADS;
    h.line_bytes  = VX_CFG_MEM_BLOCK_SIZE;
    h.bvh_width   = VX_CFG_RTU_BVH_WIDTH;
    h.ray_bytes   = sizeof(RaylogRay);
    std::fwrite(&h, sizeof(h), 1, fp_);
  }

  ~Logger() {
    if (fp_ == nullptr) return;
    std::fclose(fp_);
    std::fprintf(stderr, "[rtu-raylog] rays=%llu logged=%llu replayable=%llu lines=%zu epochs=%u\n",
                 (unsigned long long)seen_, (unsigned long long)logged_,
                 (unsigned long long)replayable_, image_.size(), epoch_ + 1);
  }

  bool on() const { return fp_ != nullptr; }

  void set_ram(const RAM* ram) {
    ram_ = ram;
  }

  void accept(const void* owner, uint32_t slot, const RtuReq& req) {
    cb_[{owner, slot}].fill(0);
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (req.tmask_bits & (1u << t)) snapshot_scene(req.scene_root[t]);
    }
  }

  void callback(const void* owner, uint32_t slot, uint32_t lane, uint32_t cb_type) {
    cb_[{owner, slot}].at(lane) |= 1u << (cb_type & 7);
  }

  void walk_done(const std::unordered_map<uint64_t, LineBuf>& lines) {
    if (full()) return;
    for (const auto& kv : lines) {
      auto it = image_.find(kv.first);
      if (it != image_.end()) {
        if (it->second == kv.second) continue;
        // The device rewrote a line the scene image already holds: the rays
        // from here on see a different memory image.
        ++epoch_;
        it->second = kv.second;
      } else {
        image_.emplace(kv.first, kv.second);
      }
      RaylogLine r{};
      r.type  = REC_LINE;
      r.epoch = epoch_;
      r.addr  = kv.first;
      std::memcpy(r.data, kv.second.data(), sizeof(r.data));
      std::fwrite(&r, sizeof(r), 1, fp_);
    }
  }

  void terminal(const void* owner, uint32_t slot, const RtuReq& req,
                const std::array<LaneState, VX_CFG_NUM_THREADS>& lanes) {
    auto& cbm = cb_[{owner, slot}];
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      const LaneState& l = lanes[t];
      if (!l.active) continue;
      uint64_t seq = seen_++;
      if (full() || (seq % every_) != 0) continue;
      RaylogRay r{};
      r.type = REC_RAY;
      const uint32_t cbs = cbm[t];
      const bool decided_by_shader = cbs & ((1u << VX_RT_CB_TYPE_ANYHIT) | (1u << VX_RT_CB_TYPE_PROC));
      r.info = (cbs << kInfoCbShift) | (decided_by_shader ? 0u : kInfoReplayable);
      r.seq        = uint32_t(seq);
      r.epoch      = epoch_;
      r.scene_root = req.scene_root[t];
      r.ray_flags  = req.flags[t];
      r.cull_mask  = req.cull_mask[t];
      r.warp_lane  = t | (req.warp_id << 8) | (slot << 16);
      r.origin[0] = bits(req.origin_x[t]); r.origin[1] = bits(req.origin_y[t]); r.origin[2] = bits(req.origin_z[t]);
      r.dir[0]    = bits(req.dir_x[t]);    r.dir[1]    = bits(req.dir_y[t]);    r.dir[2]    = bits(req.dir_z[t]);
      r.tmin = bits(req.tmin[t]);
      r.tmax = bits(req.tmax[t]);
      if (l.hit) {
        r.status      = VX_RT_STS_DONE_HIT;
        r.hit_t       = bits(l.hit_t);
        r.hit_u       = bits(l.hit_u);
        r.hit_v       = bits(l.hit_v);
        r.prim        = l.hit_prim;
        r.geom        = l.hit_geometry;
        r.inst_id     = l.hit_instance_id;
        r.inst_custom = l.hit_instance_custom;
        r.hit_attr    = l.hit_attr;
      } else {
        r.status = VX_RT_STS_DONE_MISS;
      }
      std::fwrite(&r, sizeof(r), 1, fp_);
      ++logged_;
      if (!decided_by_shader) ++replayable_;
    }
  }

private:
  bool full() const { return logged_ >= max_rays_; }

  // Instance records sit below the scene root, one stride per instance id.
  static constexpr uint64_t kInstTableSpan = 64 * 1024;
  static constexpr uint64_t kMaxSceneBytes = 1ull << 30;

  void snapshot_scene(uint32_t root) {
    if (ram_ == nullptr || full() || !snapped_.insert(root).second) return;
    const RAM& ram = *ram_;
    uint32_t scene_bytes = 0;
    for (int i = 0; i < 4; ++i) scene_bytes |= uint32_t(ram[root + 8 + i]) << (8 * i);
    const uint64_t lo = (root > kInstTableSpan ? root - kInstTableSpan : 0) & kRtuLineMask;
    const uint64_t hi = uint64_t(root) + std::min<uint64_t>(scene_bytes, kMaxSceneBytes);
    uint64_t added = 0;
    for (uint64_t a = lo; a < hi; a += VX_CFG_MEM_BLOCK_SIZE) {
      if (image_.count(a)) continue;
      LineBuf line;
      bool nonzero = false;
      for (uint32_t i = 0; i < VX_CFG_MEM_BLOCK_SIZE; ++i) {
        line[i] = ram[a + i];
        nonzero |= line[i] != 0;
      }
      if (!nonzero) continue;
      image_.emplace(a, line);
      RaylogLine r{};
      r.type  = REC_LINE;
      r.epoch = epoch_;
      r.addr  = a;
      std::memcpy(r.data, line.data(), sizeof(r.data));
      std::fwrite(&r, sizeof(r), 1, fp_);
      ++added;
    }
    std::fprintf(stderr, "[rtu-raylog] scene 0x%x: %u bytes, snapshot %llu lines\n",
                 root, scene_bytes, (unsigned long long)added);
  }

  const RAM* ram_ = nullptr;
  std::unordered_set<uint32_t> snapped_;
  FILE*      fp_ = nullptr;
  unsigned long long max_rays_ = 4ull << 20;
  unsigned long long every_    = 1;
  uint64_t   seen_       = 0;
  uint64_t   logged_     = 0;
  uint64_t   replayable_ = 0;
  uint32_t   epoch_      = 0;
  std::unordered_map<uint64_t, LineBuf> image_;
  std::map<std::pair<const void*, uint32_t>, std::array<uint32_t, VX_CFG_NUM_THREADS>> cb_;
};

Logger& logger() {
  static Logger inst;
  return inst;
}

}  // namespace

bool enabled() {
  static const bool on = logger().on();
  return on;
}

void attach_ram(const RAM* ram) {
  if (enabled()) logger().set_ram(ram);
}

void on_accept(const void* owner, uint32_t slot, const RtuReq& req) {
  logger().accept(owner, slot, req);
}

void on_callback(const void* owner, uint32_t slot, uint32_t lane, uint32_t cb_type) {
  logger().callback(owner, slot, lane, cb_type);
}

void on_walk_done(const std::unordered_map<uint64_t, LineBuf>& lines) {
  logger().walk_done(lines);
}

void on_terminal(const void* owner, uint32_t slot, const RtuReq& req,
                 const std::array<LaneState, VX_CFG_NUM_THREADS>& lanes) {
  logger().terminal(owner, slot, req, lanes);
}

}}}  // namespace vortex::rtu::raylog
