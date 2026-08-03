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

#include "firesim_sim.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <gmp.h>

#include "bridges/loadmem.h"
#include "bridges/peek_poke.h"
#include "core/simif.h"
#include "core/simulation.h"
#include "core/widget_registry.h"

// Provided by the simulator's entry translation unit.
extern int entry(int argc, char **argv);

namespace {

// Requests are closures executed on the simulator thread. Handing over a
// closure rather than a tagged struct keeps the queue from having to grow a
// case for every new operation.
using request_t = std::function<void()>;

// Target DRAM is reached through the LoadMem widget, which moves one
// fixed-width chunk at a time. Unaligned or partial edges are handled with a
// read-modify-write so callers can use arbitrary byte ranges.
void access_mem(uint64_t addr, uint64_t size, void *value, bool is_write);

struct channel_t {
  std::mutex mtx;
  std::condition_variable to_sim;
  std::condition_variable to_host;
  std::queue<request_t> pending;
  bool request_done = false;
  bool ready = false;
  bool shutdown = false;
};

class vortex_simulation_t;

// The simulator's own globals are never torn down, so these outlive the
// wrapper and are cleared only when the run loop exits.
channel_t *g_channel = nullptr;
vortex_simulation_t *g_sim = nullptr;

// A simulation whose run loop services requests instead of executing a fixed
// script. Everything the target does happens on this thread: the bridges are
// not thread-safe and must never be touched by the caller.
class vortex_simulation_t final : public simulation_t {
public:
  vortex_simulation_t(widget_registry_t &registry,
                      const std::vector<std::string> &args)
      : simulation_t(registry, args),
        peek_poke(registry.get_widget<peek_poke_t>()),
        loadmem(registry.get_widget<loadmem_t>()) {}

  int simulation_run() override {
    g_sim = this;
    {
      std::lock_guard<std::mutex> lock(g_channel->mtx);
      g_channel->ready = true;
    }
    g_channel->to_host.notify_all();

    for (;;) {
      request_t req;
      {
        std::unique_lock<std::mutex> lock(g_channel->mtx);
        g_channel->to_sim.wait(
            lock, [] { return g_channel->shutdown || !g_channel->pending.empty(); });
        if (g_channel->shutdown && g_channel->pending.empty()) {
          g_sim = nullptr;
          return 0;
        }
        req = std::move(g_channel->pending.front());
        g_channel->pending.pop();
      }

      req();

      {
        std::lock_guard<std::mutex> lock(g_channel->mtx);
        g_channel->request_done = true;
      }
      g_channel->to_host.notify_all();
    }
  }

  void step(uint32_t n) {
    peek_poke.step(n, true);
  }

  peek_poke_t &peek_poke;
  loadmem_t &loadmem;
};

} // namespace

// Replaces the TEST_MAIN-generated definition: the simulator is a library here,
// not a self-contained test.
std::unique_ptr<simulation_t>
create_simulation(simif_t &simif,
                  widget_registry_t &registry,
                  const std::vector<std::string> &args) {
  (void)simif;
  return std::make_unique<vortex_simulation_t>(registry, args);
}

namespace {

void access_mem(uint64_t addr, uint64_t size, void *value, bool is_write) {
  const size_t chunk = g_sim->loadmem.get_mem_data_chunk();
  auto *bytes = static_cast<uint8_t *>(value);

  mpz_t word;
  mpz_init(word);

  uint64_t done = 0;
  while (done < size) {
    const uint64_t cur = addr + done;
    const uint64_t base = (cur / chunk) * chunk;
    const uint64_t off = cur - base;
    const uint64_t n = std::min<uint64_t>(chunk - off, size - done);

    std::vector<uint8_t> buf(chunk, 0);
    const bool partial = (off != 0) || (n != chunk);
    if (!is_write || partial) {
      g_sim->loadmem.read_mem(base, word);
      size_t count = 0;
      mpz_export(buf.data(), &count, -1, 1, 0, 0, word);
    }

    if (is_write) {
      std::memcpy(buf.data() + off, bytes + done, n);
      mpz_import(word, chunk, -1, 1, 0, 0, buf.data());
      g_sim->loadmem.write_mem(base, word);
    } else {
      std::memcpy(bytes + done, buf.data() + off, n);
    }
    done += n;
  }

  mpz_clear(word);
}

} // namespace

namespace vortex {

class firesim_sim::Impl {
public:
  ~Impl() {
    if (thread_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(channel_.mtx);
        channel_.shutdown = true;
      }
      channel_.to_sim.notify_all();
      thread_.join();
    }
    g_channel = nullptr;
  }

  int init(const char *bitstream) {
    g_channel = &channel_;

    argv_storage_.emplace_back("firesim");
    if (bitstream != nullptr && *bitstream != '\0') {
      argv_storage_.emplace_back(std::string("+binary_file=") + bitstream);
    }
    for (auto &s : argv_storage_) {
      argv_.push_back(const_cast<char *>(s.c_str()));
    }

    thread_ = std::thread(
        [this] { entry(static_cast<int>(argv_.size()), argv_.data()); });

    std::unique_lock<std::mutex> lock(channel_.mtx);
    channel_.to_host.wait(lock, [this] { return channel_.ready; });

    // Vortex samples reset for VX_CFG_RESET_DELAY cycles before it accepts any
    // request, so clear the control inputs and hold reset well past that.
    return submit([] {
      g_sim->peek_poke.poke("ctrl_start", 0, true);
      g_sim->peek_poke.poke("ctrl_dcr_req_valid", 0, true);
      g_sim->peek_poke.poke("reset", 1, true);
      g_sim->step(32);
      g_sim->peek_poke.poke("reset", 0, true);
      g_sim->step(1);
    });
  }

  // Runs `fn` on the simulator thread and waits for it to finish. Requests are
  // serialized, so the target advances only in the order asked for.
  int submit(request_t fn) {
    {
      std::lock_guard<std::mutex> lock(channel_.mtx);
      if (channel_.shutdown) {
        return -1;
      }
      channel_.request_done = false;
      channel_.pending.push(std::move(fn));
    }
    channel_.to_sim.notify_all();

    std::unique_lock<std::mutex> lock(channel_.mtx);
    channel_.to_host.wait(lock, [this] { return channel_.request_done; });
    return 0;
  }

private:
  channel_t channel_;
  std::thread thread_;
  std::vector<std::string> argv_storage_;
  std::vector<char *> argv_;
};

firesim_sim::firesim_sim() : impl_(new Impl()) {}

firesim_sim::~firesim_sim() { delete impl_; }

int firesim_sim::init(const char *bitstream) { return impl_->init(bitstream); }

int firesim_sim::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->submit([addr, value] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_dcr_req_valid", 1, true);
    pp.poke("ctrl_dcr_req_rw", 1, true);
    pp.poke("ctrl_dcr_req_addr", addr, true);
    pp.poke("ctrl_dcr_req_data", value, true);
    g_sim->step(1);
    pp.poke("ctrl_dcr_req_valid", 0, true);
    g_sim->step(1);
  });
}

int firesim_sim::dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
  uint32_t sampled = 0;
  bool answered = false;
  int ret = impl_->submit([addr, tag, &sampled, &answered] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_dcr_req_valid", 1, true);
    pp.poke("ctrl_dcr_req_rw", 0, true);
    pp.poke("ctrl_dcr_req_addr", addr, true);
    pp.poke("ctrl_dcr_req_data", tag, true);
    g_sim->step(1);
    pp.poke("ctrl_dcr_req_valid", 0, true);

    // Bounded rather than open-ended: a target that never answers must fail
    // the read instead of hanging the runtime.
    for (int i = 0; i < 1024; ++i) {
      g_sim->step(1);
      if (pp.peek("ctrl_dcr_rsp_valid", true) != 0) {
        sampled = pp.peek("ctrl_dcr_rsp_data", true);
        answered = true;
        break;
      }
    }
  });
  if (ret != 0) {
    return ret;
  }
  if (!answered) {
    return -1;
  }
  *value = sampled;
  return 0;
}

int firesim_sim::start() {
  return impl_->submit([] {
    auto &pp = g_sim->peek_poke;
    pp.poke("ctrl_start", 1, true);
    g_sim->step(1);
    pp.poke("ctrl_start", 0, true);
    g_sim->step(1);
  });
}

int firesim_sim::is_busy(bool *busy) {
  uint32_t sampled = 0;
  // Advance before sampling: the target only moves while it is being asked to,
  // so a caller that merely polls without stepping would spin forever.
  int ret = impl_->submit([&sampled] {
    g_sim->step(64);
    sampled = g_sim->peek_poke.peek("ctrl_busy", true);
  });
  if (ret != 0) {
    return ret;
  }
  *busy = (sampled != 0);
  return 0;
}

int firesim_sim::ready_wait(int64_t timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    bool busy = true;
    int ret = this->is_busy(&busy);
    if (ret != 0) {
      return ret;
    }
    if (!busy) {
      return 0;
    }
    if (timeout_ms >= 0 && std::chrono::steady_clock::now() >= deadline) {
      return -1;
    }
  }
}

int firesim_sim::mem_write(uint64_t addr, uint64_t size, const void *value) {
  return impl_->submit(
      [addr, size, value] { access_mem(addr, size, const_cast<void *>(value), true); });
}

int firesim_sim::mem_read(uint64_t addr, uint64_t size, void *value) {
  return impl_->submit(
      [addr, size, value] { access_mem(addr, size, value, false); });
}

}
