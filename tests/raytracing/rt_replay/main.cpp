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
//
// rt_replay — host driver.
//
// Loads a SimX RTU ray log (VX_RTU_RAYLOG), restores every scene line the
// recorded walks read at its original device address (absolute pointers inside
// the scene stay valid), re-traces the recorded rays in batches and compares
// each result bit-exactly with the recorded one.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

namespace {

const char* kernel_file = "kernel.vxbin";

vx_device_h device  = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
vx_buffer_h rays_buf = nullptr;
vx_buffer_h res_buf  = nullptr;
std::vector<vx_buffer_h> scene_bufs;

void cleanup() {
  if (!device) return;
  for (auto b : scene_bufs) vx_buffer_release(b);
  if (rays_buf) vx_buffer_release(rays_buf);
  if (res_buf)  vx_buffer_release(res_buf);
  if (kernel)   vx_kernel_release(kernel);
  if (module_)  vx_module_release(module_);
  if (queue)    vx_queue_release(queue);
  vx_device_release(device);
  device = nullptr;
}

struct Line {
  uint64_t addr;
  uint32_t epoch;
  std::vector<uint8_t> data;
};

struct Range {
  uint64_t base;
  uint64_t size;
  vx_buffer_h buf;
};

const char* log_path   = nullptr;
const char* list_path  = nullptr;
const char* out_path   = nullptr;
uint32_t    batch      = 4096;
uint64_t    range_lo   = 0;
uint64_t    range_hi   = UINT64_MAX;
uint64_t    stride     = 1;
int64_t     only_epoch = -1;
uint32_t    max_print  = 20;
bool        all_rays   = false;
bool        verbose    = false;

void show_usage() {
  printf("Usage: rt_replay -f <raylog> [-b batch] [-r lo:hi] [-l listfile] [-s stride]\n"
         "                 [-e epoch] [-m maxprint] [-o mismatch_file] [-a] [-v]\n"
         "  -r lo:hi   replay ray indices [lo, hi) (index = RAY record order in the log)\n"
         "  -l file    replay only the ray indices listed (one per line, '#' comments;\n"
         "             a mismatch file from -o is accepted as is)\n"
         "  -s n       replay every n-th selected ray\n"
         "  -a         also replay rays a shader decided (any-hit / intersection)\n"
         "Build with the CONFIGS that recorded the log. For rtlsim also pass\n"
         "-DVX_DBG_STALL_TIMEOUT=2000000000: one hard ray can hold every warp in\n"
         "vx_rt_wait past the scheduler's default all-warps-stalled watchdog.\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "f:b:r:l:s:e:m:o:avh")) != -1) {
    switch (c) {
    case 'f': log_path = optarg; break;
    case 'b': batch = std::max(1u, (uint32_t)strtoul(optarg, nullptr, 0)); break;
    case 'r': {
      const char* colon = strchr(optarg, ':');
      range_lo = strtoull(optarg, nullptr, 0);
      if (colon && colon[1]) range_hi = strtoull(colon + 1, nullptr, 0);
      else if (!colon) range_hi = range_lo + 1;
      break;
    }
    case 'l': list_path = optarg; break;
    case 's': stride = std::max<uint64_t>(1, strtoull(optarg, nullptr, 0)); break;
    case 'e': only_epoch = strtoll(optarg, nullptr, 0); break;
    case 'm': max_print = strtoul(optarg, nullptr, 0); break;
    case 'o': out_path = optarg; break;
    case 'a': all_rays = true; break;
    case 'v': verbose = true; break;
    default: show_usage(); exit(c == 'h' ? 0 : -1);
    }
  }
  if (!log_path) { show_usage(); exit(-1); }
}

float fbits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float near_t(float t) { return t + std::fabs(t) * 0x1p-19f; }

// Why a replayed ray disagrees with its record.
const char* classify(const RaylogRay& ref, const replay_result_t& got) {
  if (got.status & REPLAY_STS_YIELDED) return "yielded";
  const bool rh = ref.status == VX_RT_STS_DONE_HIT;
  const bool gh = got.status == VX_RT_STS_DONE_HIT;
  if (rh && !gh) return "ref_hit_got_miss";
  if (!rh && gh) return "ref_miss_got_hit";
  if (!rh) return "status";
  const bool same_prim = ref.prim == got.prim && ref.inst_id == got.inst_id
      && (ref.geom & VX_RT_HIT_GEOMETRY_MASK) == (got.geom & VX_RT_HIT_GEOMETRY_MASK);
  if (same_prim) {
    if (ref.hit_t != got.t || ref.hit_u != got.u || ref.hit_v != got.v) return "same_prim_bits";
    return "attr";
  }
  if (ref.ray_flags & VX_RT_FLAG_TERMINATE_ON_FIRST_HIT) return "tofh_other_hit";
  const float a = fbits(ref.hit_t), b = fbits(got.t);
  if (a == b) return "tie_exact";
  if ((a < b && near_t(a) >= b) || (b < a && near_t(b) >= a)) return "near_tie";
  return b < a ? "got_nearer_hit" : "got_farther_hit";
}

bool matches(const RaylogRay& ref, const replay_result_t& got) {
  if (got.status != ref.status) return false;
  if (ref.status != VX_RT_STS_DONE_HIT) return true;
  return got.t == ref.hit_t && got.u == ref.hit_u && got.v == ref.hit_v
      && got.prim == ref.prim && got.geom == ref.geom
      && got.inst_id == ref.inst_id && got.inst_custom == ref.inst_custom;
}

void print_ray(FILE* fp, uint64_t idx, const RaylogRay& r, const replay_result_t& g, const char* cat) {
  fprintf(fp, "%llu seq=%u cat=%s epoch=%u scene=0x%x flags=0x%x cull=0x%x"
              " o=(%a,%a,%a) d=(%a,%a,%a) tmin=%a tmax=%a\n",
          (unsigned long long)idx, r.seq, cat, r.epoch, r.scene_root, r.ray_flags, r.cull_mask,
          fbits(r.origin[0]), fbits(r.origin[1]), fbits(r.origin[2]),
          fbits(r.dir[0]), fbits(r.dir[1]), fbits(r.dir[2]),
          fbits(r.tmin), fbits(r.tmax));
  fprintf(fp, "#   ref: sts=%u t=%a (0x%08x) u=0x%08x v=0x%08x prim=%u geom=0x%08x inst=%u custom=%u\n",
          r.status, fbits(r.hit_t), r.hit_t, r.hit_u, r.hit_v, r.prim, r.geom, r.inst_id, r.inst_custom);
  fprintf(fp, "#   got: sts=%u t=%a (0x%08x) u=0x%08x v=0x%08x prim=%u geom=0x%08x inst=%u custom=%u\n",
          g.status, fbits(g.t), g.t, g.u, g.v, g.prim, g.geom, g.inst_id, g.inst_custom);
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // ── load the log ──────────────────────────────────────────────────
  FILE* fp = fopen(log_path, "rb");
  if (!fp) { printf("Error: cannot open %s\n", log_path); return -1; }
  RaylogHeader hdr{};
  if (fread(&hdr, sizeof(hdr), 1, fp) != 1 || hdr.magic != RAYLOG_MAGIC
      || hdr.version != RAYLOG_VERSION || hdr.ray_bytes != sizeof(RaylogRay)) {
    printf("Error: %s is not a version-%u ray log\n", log_path, RAYLOG_VERSION);
    return -1;
  }
#ifdef VX_CFG_RTU_BVH_WIDTH
  if (hdr.bvh_width != VX_CFG_RTU_BVH_WIDTH) {
    printf("Error: log recorded with RTU_BVH_WIDTH=%u, replay built with %u\n",
           hdr.bvh_width, (uint32_t)VX_CFG_RTU_BVH_WIDTH);
    return -1;
  }
#endif
  const uint32_t lb = hdr.line_bytes;
  std::vector<Line> lines;
  std::vector<RaylogRay> rays;
  {
    std::vector<uint8_t> rec(std::max<size_t>(sizeof(RaylogRay), 16 + lb));
    uint32_t type;
    while (fread(&type, 4, 1, fp) == 1) {
      if (type == RAYLOG_REC_LINE) {
        if (fread(rec.data() + 4, 12 + lb, 1, fp) != 1) break;
        Line l;
        memcpy(&l.epoch, rec.data() + 4, 4);
        memcpy(&l.addr, rec.data() + 8, 8);
        l.data.assign(rec.data() + 16, rec.data() + 16 + lb);
        lines.push_back(std::move(l));
      } else if (type == RAYLOG_REC_RAY) {
        RaylogRay r;
        r.type = type;
        if (fread(reinterpret_cast<uint8_t*>(&r) + 4, sizeof(r) - 4, 1, fp) != 1) break;
        rays.push_back(r);
      } else {
        printf("Error: corrupt record type %u in %s\n", type, log_path);
        return -1;
      }
    }
  }
  fclose(fp);
  uint32_t num_epochs = 1;
  for (auto& l : lines) num_epochs = std::max(num_epochs, l.epoch + 1);
  printf("raylog: %zu rays, %zu lines (%u B), %u epoch(s), bvh_width=%u, recorded NT=%u\n",
         rays.size(), lines.size(), lb, num_epochs, hdr.bvh_width, hdr.num_threads);

  // ── select rays ───────────────────────────────────────────────────
  std::vector<uint64_t> sel;
  {
    std::vector<uint64_t> cand;
    if (list_path) {
      FILE* lf = fopen(list_path, "r");
      if (!lf) { printf("Error: cannot open %s\n", list_path); return -1; }
      char buf[4096];
      while (fgets(buf, sizeof(buf), lf)) {
        if (buf[0] == '#' || buf[0] == '\n') continue;
        cand.push_back(strtoull(buf, nullptr, 0));
      }
      fclose(lf);
    } else {
      for (uint64_t i = range_lo; i < std::min<uint64_t>(range_hi, rays.size()); ++i) cand.push_back(i);
    }
    uint64_t skipped_cb = 0, n = 0;
    for (uint64_t i : cand) {
      if (i >= rays.size()) continue;
      if (range_lo > i || i >= range_hi) continue;
      const RaylogRay& r = rays[i];
      if (only_epoch >= 0 && r.epoch != (uint32_t)only_epoch) continue;
      if (!all_rays && !(r.info & RAYLOG_INFO_REPLAYABLE)) { ++skipped_cb; continue; }
      if ((n++ % stride) != 0) continue;
      sel.push_back(i);
    }
    std::stable_sort(sel.begin(), sel.end(), [&](uint64_t a, uint64_t b) {
      return rays[a].epoch < rays[b].epoch;
    });
    printf("selected %zu rays (%llu skipped: decided by a shader callback)\n",
           sel.size(), (unsigned long long)skipped_cb);
  }
  if (sel.empty()) { printf("nothing to replay\n"); return 0; }

  // ── device + scene reservation ────────────────────────────────────
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Claim the recorded scene addresses before anything else is allocated, in
  // page-aligned ranges (nearby pages merged to keep the buffer count low).
  std::vector<Range> ranges;
  {
    const uint64_t page = 4096, merge_gap = 64 * 1024;
    std::vector<uint64_t> pages;
    for (auto& l : lines) {
      for (uint64_t p = l.addr & ~(page - 1); p < l.addr + lb; p += page) pages.push_back(p);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (uint64_t p : pages) {
      if (!ranges.empty() && p <= ranges.back().base + ranges.back().size + merge_gap) {
        ranges.back().size = p + page - ranges.back().base;
      } else {
        ranges.push_back({p, page, nullptr});
      }
    }
    uint64_t total = 0;
    for (auto& r : ranges) {
      RT_CHECK(vx_buffer_reserve(device, r.base, r.size, VX_MEM_READ, &r.buf));
      scene_bufs.push_back(r.buf);
      total += r.size;
    }
    printf("scene: %zu reserved range(s), %llu KB\n", ranges.size(), (unsigned long long)(total >> 10));
  }

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  const uint32_t NT = VX_CFG_NUM_THREADS;
  uint64_t cap = 0;

  std::map<std::string, uint64_t> cats;
  uint64_t replayed = 0, mismatched = 0, printed = 0;
  FILE* of = out_path ? fopen(out_path, "w") : nullptr;
  if (of) fprintf(of, "# rt_replay mismatches of %s: <ray index> then ref/got detail\n", log_path);
  double kernel_s = 0;
  int32_t loaded_epoch = -1;

  for (size_t pos = 0; pos < sel.size();) {
    // A batch never spans epochs: each epoch replays against its own image.
    const uint32_t epoch = rays[sel[pos]].epoch;
    size_t end = pos;
    while (end < sel.size() && end - pos < batch && rays[sel[end]].epoch == epoch) ++end;

    if ((int32_t)epoch != loaded_epoch) {
      // Image of `epoch`: per address, the newest version recorded at or before
      // it, else the first one recorded (the line was not rewritten before then).
      std::map<uint64_t, const Line*> img;
      for (auto& l : lines) {
        auto it = img.find(l.addr);
        if (it == img.end()) { img[l.addr] = &l; continue; }
        if (l.epoch <= epoch && (it->second->epoch > epoch || l.epoch >= it->second->epoch)) it->second = &l;
      }
      std::vector<std::vector<uint8_t>> hosts;   // alive until the queue drains
      for (auto& r : ranges) {
        hosts.emplace_back(r.size, 0);
        auto& host = hosts.back();
        for (auto it = img.lower_bound(r.base); it != img.end() && it->first < r.base + r.size; ++it) {
          uint64_t off = it->first - r.base;
          memcpy(host.data() + off, it->second->data.data(), std::min<uint64_t>(lb, r.size - off));
        }
        RT_CHECK(vx_enqueue_write(queue, r.buf, 0, host.data(), r.size, 0, nullptr, nullptr));
      }
      RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
      loaded_epoch = epoch;
    }

    // Group by the warp-uniform trace config; pad each group to whole warps
    // with copies of its last ray (their results are dropped).
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::vector<uint64_t>> groups;
    for (size_t k = pos; k < end; ++k) {
      const RaylogRay& r = rays[sel[k]];
      groups[{r.scene_root, r.ray_flags & ~(VX_RT_FLAG_ENABLE_CHS | VX_RT_FLAG_ENABLE_MISS), r.cull_mask}].push_back(sel[k]);
    }
    std::vector<replay_ray_t> dev_rays;
    std::vector<int64_t> owner;   // ray index, or -1 for padding
    for (auto& g : groups) {
      auto& v = g.second;
      size_t padded = (v.size() + NT - 1) / NT * NT;
      for (size_t k = 0; k < padded; ++k) {
        uint64_t idx = v[std::min(k, v.size() - 1)];
        const RaylogRay& r = rays[idx];
        replay_ray_t d{};
        d.scene = r.scene_root;
        d.flags = std::get<1>(g.first);
        d.cull  = r.cull_mask;
        for (int j = 0; j < 3; ++j) { d.origin[j] = fbits(r.origin[j]); d.dir[j] = fbits(r.dir[j]); }
        d.tmin = fbits(r.tmin);
        d.tmax = fbits(r.tmax);
        dev_rays.push_back(d);
        owner.push_back(k < v.size() ? (int64_t)idx : -1);
      }
    }
    const uint32_t count = (uint32_t)dev_rays.size();
    if (count > cap) {
      if (rays_buf) vx_buffer_release(rays_buf);
      if (res_buf)  vx_buffer_release(res_buf);
      cap = std::max<uint64_t>(count, batch + NT);
      RT_CHECK(vx_buffer_create(device, cap * sizeof(replay_ray_t), VX_MEM_READ, &rays_buf));
      RT_CHECK(vx_buffer_create(device, cap * sizeof(replay_result_t), VX_MEM_WRITE, &res_buf));
    }
    kernel_arg_t arg{};
    RT_CHECK(vx_buffer_address(rays_buf, &arg.rays_addr));
    RT_CHECK(vx_buffer_address(res_buf, &arg.results_addr));
    arg.count = count;
    RT_CHECK(vx_enqueue_write(queue, rays_buf, 0, dev_rays.data(), count * sizeof(replay_ray_t), 0, nullptr, nullptr));
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));

    if (verbose) {
      printf("launch [%zu, %zu): %u threads\n", pos, end, count);
      fflush(stdout);
    }
    auto t0 = std::chrono::steady_clock::now();
    vx_event_h lev = nullptr, rev = nullptr;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 1;
    li.grid_dim[0]  = count / NT;
    li.block_dim[0] = NT;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &lev));
    std::vector<replay_result_t> res(count);
    RT_CHECK(vx_enqueue_read(queue, res.data(), res_buf, 0, count * sizeof(replay_result_t), 1, &lev, &rev));
    RT_CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rev);
    vx_event_release(lev);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    kernel_s += dt;

    uint64_t bad = 0;
    for (uint32_t k = 0; k < count; ++k) {
      if (owner[k] < 0) continue;
      const RaylogRay& r = rays[owner[k]];
      ++replayed;
      if (matches(r, res[k])) continue;
      ++bad;
      const char* cat = classify(r, res[k]);
      ++cats[cat];
      if (of) print_ray(of, owner[k], r, res[k], cat);
      if (printed < max_print) { print_ray(stdout, owner[k], r, res[k], cat); ++printed; }
    }
    mismatched += bad;
    if (of) fflush(of);   // a model that hangs or aborts later keeps what it reported
    if (verbose || sel.size() > batch) {
      printf("batch [%zu, %zu) epoch %u: %llu rays, %llu mismatches, %.1fs\n",
             pos, end, epoch, (unsigned long long)(end - pos), (unsigned long long)bad, dt);
      fflush(stdout);
    }
    pos = end;
  }
  if (of) fclose(of);

  printf("replayed %llu rays in %.1fs: %llu match, %llu mismatch\n",
         (unsigned long long)replayed, kernel_s,
         (unsigned long long)(replayed - mismatched), (unsigned long long)mismatched);
  for (auto& c : cats) printf("  %-18s %llu\n", c.first.c_str(), (unsigned long long)c.second);

  cleanup();
  if (mismatched) {
    printf("FAILED!\n");
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
