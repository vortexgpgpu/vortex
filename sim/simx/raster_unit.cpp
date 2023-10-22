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

#include "raster_unit.h"
#include "graphics.h"
#include <VX_config.h>
#include "mempool.h"
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "mem.h"

using namespace vortex;

#define STAMP_POOL_MAX_SIZE   1024

struct prim_mem_trace_t {
  uint64_t              prim_addr;
  std::vector<uint64_t> edge_addrs;
  uint32_t              stamps;
};

struct tile_mem_trace_t {
  std::vector<uint64_t>       header_addrs;
  std::list<prim_mem_trace_t> primitives;
  bool end_of_tile;
};

namespace std {
template <unsigned F>
std::ostream& operator<<(std::ostream &out, const cocogfx::TFixed<F>& value) {
  out << value.data();
  return out;
}
}

class Rasterizer : public graphics::Rasterizer {
public:

  struct Stamp {
    uint32_t pos_mask;
    std::array<graphics::vec3e_t, 4> bcoords;
    uint32_t pid;
    Stamp* next_;    
    Stamp* prev_;

    Stamp(uint32_t pos_mask, graphics::vec3e_t bcoords[4], uint32_t  pid) 
      : pos_mask(pos_mask)
      , bcoords({bcoords[0], bcoords[1], bcoords[2], bcoords[3]})
      , pid(pid)
      , next_(nullptr)
      , prev_(nullptr) 
    {}

    void* operator new(size_t /*size*/) {
      return allocator().allocate();
    }

    void operator delete(void* ptr) {
      allocator().deallocate(ptr);
    }

  private:

    static MemoryPool<Stamp>& allocator() {
      static MemoryPool<Stamp> instance(1024);
      return instance;
    }
  };

  Rasterizer(uint32_t raster_index,              
             uint32_t raster_count,
             uint32_t tile_logsize, 
             uint32_t block_logsize) 
    : graphics::Rasterizer(shaderFunctionCB, this, tile_logsize, block_logsize)
    , raster_index_(raster_index)
    , raster_count_(raster_count)
    , stamps_head_(nullptr)
    , stamps_tail_(nullptr)
    , stamps_size_(0)
    , done_(false) {
    assert(block_logsize >= 1);
    assert(tile_logsize >= block_logsize);
  }

  ~Rasterizer() {} 
  
  uint32_t id() const {
    return raster_index_;
  }

  void configure(const graphics::RasterDCRS& dcrs) {
    // get device configuration    
    graphics::Rasterizer::configure(dcrs);
    num_tiles_     = dcrs.read(VX_DCR_RASTER_TILE_COUNT);
    tbuf_baseaddr_ = uint64_t(dcrs.read(VX_DCR_RASTER_TBUF_ADDR)) << 6;
    pbuf_baseaddr_ = uint64_t(dcrs.read(VX_DCR_RASTER_PBUF_ADDR)) << 6;
    pbuf_stride_   = dcrs.read(VX_DCR_RASTER_PBUF_STRIDE);

    tbuf_addr_  = tbuf_baseaddr_ + raster_index_ * sizeof(graphics::rast_tile_header_t);
    cur_tile_   = raster_index_;
    done_       = (cur_tile_ >= num_tiles_);
    cur_prim_   = 0;
    pids_count_ = 0;
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  Stamp* fetch() {
    while (!done_ && stamps_head_ == nullptr) {
      this->renderNextPrimitive();
    }
    return this->dequeue_stamp();
  }
  
  bool done() const {
    return done_ && (stamps_head_ == nullptr);
  }

  auto& mem_traces() {
    return mem_traces_;
  }

  auto& mem_traces() const {
    return mem_traces_;
  }

private:

  void renderNextPrimitive() {  
    if (done_)
      return;  
    if (0 == pids_count_) {
      mem_traces_.push_back({});      
      auto& mem_trace = mem_traces_.back();
      mem_trace.end_of_tile = false;

      uint32_t tile_xy;
      uint32_t prim_header;

      // read next tile header from tile buffer
      mem_->read(&tile_xy, tbuf_addr_, 4);
      mem_trace.header_addrs.push_back(tbuf_addr_);
      tile_x_ = (tile_xy & 0xffff) << tile_logsize_;
      tile_y_ = (tile_xy >> 16) << tile_logsize_;
      //printf("*** raster%d-mem: add=%d, tile_x=%d, tile_y=%d\n", raster_index_, tbuf_addr_, tile_x_, tile_y_);
      tbuf_addr_ += 4;
      
      mem_->read(&prim_header, tbuf_addr_, 4);
      mem_trace.header_addrs.push_back(tbuf_addr_);
      pids_offset_ = (prim_header & 0xffff);
      pids_count_  = (prim_header >> 16);
      //printf("*** raster%d-mem: add=%d, prim_off=%d, prim_cnt=%d\n", raster_index_, tbuf_addr_, pids_offset_, pids_count_);
      tbuf_addr_ += 4;

      assert(pids_count_ > 0);      
      cur_prim_ = 0;
    }

    assert(!mem_traces_.empty());
    auto& mem_trace = mem_traces_.back();
    prim_mem_trace_t prim_trace;

    // read next primitive index from tile buffer
    prim_trace.prim_addr = tbuf_addr_ + (pids_offset_ << 2);
    mem_->read(&pid_, prim_trace.prim_addr, 4);    
    //printf("*** raster%d-mem: add=%d, pid=%d\n", raster_index_, prim_trace.prim_addr, pid_);
    ++pids_offset_;

    uint32_t x = tile_x_;
    uint32_t y = tile_y_;

    //printf("*** raster%d-primitive: tile=%d/%d, prim=%d/%d, pid=%d, tx=%d, ty=%d\n", raster_index_, cur_tile_, num_tiles_, cur_prim_, pids_count_, pid_, x, y);

    // get primitive edges
    graphics::vec3e_t edges[3];
    auto pbuf_addr = pbuf_baseaddr_ + pid_ * pbuf_stride_;
    for (int i = 0; i < 3; ++i) {
      mem_->read(&edges[i].x, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.x=%d\n", raster_index_, pbuf_addr, edges[i].x.data());      
      pbuf_addr += 4;
      
      mem_->read(&edges[i].y, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.y=%d\n", raster_index_, pbuf_addr, edges[i].y.data());
      pbuf_addr += 4;
      
      mem_->read(&edges[i].z, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.z=%d\n", raster_index_, pbuf_addr, edges[i].z.data());
      pbuf_addr += 4;      
    }

    /*printf("*** raster%d-edges={{0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}}\n", 
      raster_index_, 
      edges[0].x.data(), edges[0].y.data(), edges[0].z.data(),
      edges[1].x.data(), edges[1].y.data(), edges[1].z.data(),
      edges[2].x.data(), edges[2].y.data(), edges[2].z.data());*/

    // Render the primitive
    this->renderPrimitive(x, y, pid_, edges);

    // printf("*** raster%d: generated %d stamps\n", raster_index_, stamps_size_);
    prim_trace.stamps = stamps_size_;
    mem_trace.primitives.push_back(prim_trace);

    // Move to next primitive
    ++cur_prim_;
    if (cur_prim_ == pids_count_) {
      mem_trace.end_of_tile = true;
      // Move to next tile      
      cur_tile_  += raster_count_;      
      tbuf_addr_ += (raster_count_-1) * sizeof(graphics::rast_tile_header_t);
      pids_count_ = 0;
      done_       = (cur_tile_ >= num_tiles_);
    }
  }

  void enqueue_stamp(uint32_t pos_mask, graphics::vec3e_t bcoords[4], uint32_t pid) {
    auto stamp = new Stamp(pos_mask, bcoords, pid);
    stamp->next_ = stamps_tail_;
    stamp->prev_ = nullptr;
    if (stamps_tail_)
      stamps_tail_->prev_ = stamp;
    else
      stamps_head_ = stamp;
    stamps_tail_ = stamp;
    ++stamps_size_;
  }

  Stamp* dequeue_stamp() {
    auto stamp = stamps_head_;
    if (stamp != nullptr) {    
      stamps_head_ = stamp->prev_;
      if (stamps_head_)
        stamps_head_->next_ = nullptr;
      else
        stamps_tail_ = nullptr;
      --stamps_size_;
    }
    return stamp;
  }

  static void shaderFunctionCB(    
    uint32_t  pos_mask,
    graphics::vec3e_t bcoords[4],
    uint32_t  pid,
    void*     cb_arg) {
    reinterpret_cast<Rasterizer*>(cb_arg)->enqueue_stamp(pos_mask, bcoords, pid);
  }
      
  uint32_t raster_index_;
  uint32_t raster_count_;
  RAM*     mem_;  
  uint32_t num_tiles_;
  uint64_t tbuf_baseaddr_;    
  uint64_t pbuf_baseaddr_;
  uint32_t pbuf_stride_;
  uint64_t tbuf_addr_;
  uint32_t tile_x_;
  uint32_t tile_y_;
  uint32_t pids_offset_;
  uint32_t pids_count_;    
  uint32_t cur_tile_;
  uint32_t cur_prim_;
  uint32_t pid_;
  Stamp*   stamps_head_;
  Stamp*   stamps_tail_;
  uint32_t stamps_size_;
  std::list<tile_mem_trace_t> mem_traces_;
  bool     done_;
};

///////////////////////////////////////////////////////////////////////////////

class RasterUnit::Impl {
public:

  Impl(RasterUnit* simobject,     
       uint32_t raster_index,
       uint32_t raster_count,
       const Arch &arch,
       const DCRS& dcrs,       
       const Config& config) 
    : simobject_(simobject)  
    , arch_(arch)
    , dcrs_(dcrs)
    , rasterizer_(raster_index, raster_count, config.tile_logsize, config.block_logsize)
    , pending_reqs_(RASTER_MEM_QUEUE_SIZE)    
    , mem_trace_state_(e_mem_trace_state::header)
  {}

  ~Impl() {}

  uint32_t id() const {
    return rasterizer_.id();
  }

  void reset() {
    rasterizer_.configure(dcrs_);
    mem_trace_state_ = e_mem_trace_state::header;
    fetched_stamps_ = 0;
  }

  void tick() {
    // check input queue
    if (!simobject_->Input.empty()) {
      auto trace = simobject_->Input.front();
      auto raster_done = rasterizer_.done() 
                      && rasterizer_.mem_traces().empty() 
                      && stamps_.empty();

      if (raster_done) {
        // no more stamps
        simobject_->Output.send(trace, 1);
        simobject_->Input.pop();
      } else {
        // fetch stamps to service each request 
        // the request size is the current number of active threads
        auto num_threads = trace->tmask.count();
        uint32_t fetched_stamps = fetched_stamps_;
        while (fetched_stamps < num_threads) {
          if (stamps_.empty())
            break;
          auto count = stamps_.front();
          fetched_stamps += count;
          stamps_.pop();
        }
        if (fetched_stamps >= num_threads) {
          fetched_stamps -= num_threads;
          simobject_->Output.send(trace, 1);
          simobject_->Input.pop();
        }
        fetched_stamps_ = fetched_stamps;
      }                  
    }

    // process memory traces
    auto& mem_traces = rasterizer_.mem_traces();
    if (!simobject_->MemRsps.empty()) {
      assert(!mem_traces.empty());
      auto& mem_rsp = simobject_->MemRsps.front();
      auto& entry = pending_reqs_.at(mem_rsp.tag);
      assert(entry.count);
      --entry.count; // track remaining addresses 
      if (0 == entry.count) {
        switch (mem_trace_state_) {
        case e_mem_trace_state::header: {
          mem_trace_state_ = e_mem_trace_state::primitive;
        } break;
        case e_mem_trace_state::primitive: {
          mem_trace_state_ = e_mem_trace_state::edges;
        } break;
        case e_mem_trace_state::edges: {          
          auto& mem_trace = mem_traces.front();
          auto& primitive = mem_trace.primitives.front();

          stamps_.push(primitive.stamps);
          
          mem_trace.primitives.pop_front();
          if (mem_trace.primitives.empty() && mem_trace.end_of_tile) {
            mem_trace_state_ = e_mem_trace_state::header;
            mem_traces.pop_front();
          } else {
            mem_trace_state_ = e_mem_trace_state::primitive;
          }          
        } break; 
        default:
          break; 
        }
        pending_reqs_.release(mem_rsp.tag);
      }
      simobject_->MemRsps.pop();
    }

    for (int i = 0, n = pending_reqs_.size(); i < n; ++i) {
      if (pending_reqs_.contains(i))
        perf_stats_.latency += pending_reqs_.at(i).count;
    }

    perf_stats_.stalls += (SimPlatform::instance().cycles() > simobject_->Output.arrival_time());

    if (mem_traces.empty())
      return;

    // check pending queue is empty    
    if (!pending_reqs_.empty())          
      return;

    auto& mem_trace = mem_traces.front();

    std::vector<uint64_t> addresses;

    switch (mem_trace_state_) {
    case e_mem_trace_state::header: {
      addresses = mem_trace.header_addrs;
    } break;
    case e_mem_trace_state::primitive: {
      if (!mem_trace.primitives.empty()) {
        auto& primitive = mem_trace.primitives.front();
        addresses.push_back(primitive.prim_addr);
      }
    } break;
    case e_mem_trace_state::edges: {
      if (!mem_trace.primitives.empty()) {    
        auto& primitive = mem_trace.primitives.front();
        addresses = primitive.edge_addrs;
      }
    } break; 
    default:
      break; 
    }

    if (addresses.empty())
      return;

    auto tag = pending_reqs_.allocate({(uint32_t)addresses.size()});
    for (auto addr : addresses) {
      MemReq mem_req;
      mem_req.addr  = addr;
      mem_req.write = false;
      mem_req.tag   = tag;
      mem_req.cid   = 0;
      mem_req.uuid  = 0;
      simobject_->MemReqs.send(mem_req, 2);
      ++perf_stats_.reads;
    }
  }

  void attach_ram(RAM* mem) {
    rasterizer_.attach_ram(mem);
  } 

  uint32_t fetch(uint32_t cid, uint32_t wid, uint32_t tid, CSRs& csrs) {
    __unused (cid, wid, tid);
    auto stamp = rasterizer_.fetch();
    if (nullptr == stamp)
      return 0;

    // update CSRs
    csrs[VX_CSR_RASTER_POS_MASK]  = stamp->pos_mask;
    csrs[VX_CSR_RASTER_BCOORD_X0] = *(uint32_t*)&stamp->bcoords[0].x;
    csrs[VX_CSR_RASTER_BCOORD_Y0] = *(uint32_t*)&stamp->bcoords[0].y;
    csrs[VX_CSR_RASTER_BCOORD_Z0] = *(uint32_t*)&stamp->bcoords[0].z;
    csrs[VX_CSR_RASTER_BCOORD_X1] = *(uint32_t*)&stamp->bcoords[1].x;
    csrs[VX_CSR_RASTER_BCOORD_Y1] = *(uint32_t*)&stamp->bcoords[1].y;
    csrs[VX_CSR_RASTER_BCOORD_Z1] = *(uint32_t*)&stamp->bcoords[1].z;
    csrs[VX_CSR_RASTER_BCOORD_X2] = *(uint32_t*)&stamp->bcoords[2].x;
    csrs[VX_CSR_RASTER_BCOORD_Y2] = *(uint32_t*)&stamp->bcoords[2].y;    
    csrs[VX_CSR_RASTER_BCOORD_Z2] = *(uint32_t*)&stamp->bcoords[2].z;
    csrs[VX_CSR_RASTER_BCOORD_X3] = *(uint32_t*)&stamp->bcoords[3].x;
    csrs[VX_CSR_RASTER_BCOORD_Y3] = *(uint32_t*)&stamp->bcoords[3].y;
    csrs[VX_CSR_RASTER_BCOORD_Z3] = *(uint32_t*)&stamp->bcoords[3].z;

    return (stamp->pid << 1) | 1;
  }

  const PerfStats& perf_stats() const { 
    return perf_stats_; 
  }

private: 

  enum class e_mem_trace_state {
    header,
    primitive,
    edges
  };

  struct pending_req_t {
    uint32_t count;
  };

  RasterUnit* simobject_;        
  const Arch& arch_;
  const DCRS& dcrs_;
  Rasterizer rasterizer_;  
  std::unordered_map<uint32_t, Rasterizer::Stamp*> csrs_;
  std::queue<uint32_t> stamps_;  
  uint32_t fetched_stamps_;
  HashTable<pending_req_t> pending_reqs_;
  e_mem_trace_state mem_trace_state_;
  PerfStats perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

RasterUnit::RasterUnit(const SimContext& ctx, 
                       const char* name,     
                       uint32_t index,                   
                       uint32_t cores_per_unit,
                       const Arch &arch, 
                       const DCRS& dcrs,
                       const Config& config) 
  : SimObject<RasterUnit>(ctx, name)
  , MemReqs(this)
  , MemRsps(this)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, index, cores_per_unit, arch, dcrs, config)) 
{}

RasterUnit::~RasterUnit() {
  delete impl_;
}

void RasterUnit::reset() {
  impl_->reset();
}

void RasterUnit::tick() {
  impl_->tick();
}

uint32_t RasterUnit::id() const {
  return impl_->id();
}

void RasterUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

uint32_t RasterUnit::fetch(uint32_t cid, uint32_t wid, uint32_t tid, CSRs& csrs) {
  return impl_->fetch(cid, wid, tid, csrs);
}

const RasterUnit::PerfStats& RasterUnit::perf_stats() const {
  return impl_->perf_stats();
}
