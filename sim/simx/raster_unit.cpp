#include "raster_unit.h"
#include <VX_config.h>
#include "mempool.h"
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "mem.h"

using namespace vortex;

#define STAMP_POOL_MAX_SIZE   1024

using fixed16_t = cocogfx::TFixed<16>;

using vec2_fx_t = cocogfx::TVector2<fixed16_t>;
using vec3_fx_t = cocogfx::TVector3<fixed16_t>;

struct primitive_t {
  vec3_fx_t edges[3];
  fixed16_t extents[3];
};

static fixed16_t fxZero(0);

// Evaluate edge function
static fixed16_t evalEdgeFunction(const vec3_fx_t& e, uint32_t x, uint32_t y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// Calculate the edge extents for square corners
static fixed16_t calcEdgeExtents(const vec3_fx_t& e) {
  vec2_fx_t corners[4] = {{fxZero, fxZero},  // 00
                          {e.x,    fxZero},  // 10
                          {fxZero, e.y},     // 01
                          {e.x,    e.y}};    // 11
  auto i = (e.y >= fxZero) ? ((e.x >= fxZero) ? 3 : 2) : (e.x >= fxZero) ? 1 : 0;
  return corners[i].x + corners[i].y;
}

struct prim_mem_trace_t {
  uint32_t              prim_addr;
  std::vector<uint32_t> edge_addrs;
  uint32_t              stamps;
};

struct tile_mem_trace_t {
  std::vector<uint32_t>       header_addrs;
  std::list<prim_mem_trace_t> primitives;
  bool end_of_tile;
};

class Rasterizer {
private:    
  uint32_t index_;
  const Arch& arch_;
  const RasterUnit::DCRS& dcrs_;
  RAM* mem_;  
  uint32_t tile_logsize_;
  uint32_t block_logsize_;    
  uint32_t num_tiles_;
  uint32_t tbuf_baseaddr_;    
  uint32_t pbuf_baseaddr_;
  uint32_t pbuf_stride_;
  uint32_t tbuf_addr_;
  uint32_t dst_width_;
  uint32_t dst_height_;
  uint32_t tile_x_;
  uint32_t tile_y_;
  uint32_t prim_offset_;
  uint32_t prim_count_;    
  uint32_t cur_tile_;
  uint32_t cur_prim_;
  uint32_t pid_;
  RasterUnit::Stamp* stamps_head_;
  RasterUnit::Stamp *stamps_tail_;
  uint32_t           stamps_size_;
  std::list<tile_mem_trace_t> mem_traces_;
  bool done_;

  void stamps_push(RasterUnit::Stamp* stamp) {
    assert(stamp);
    stamp->next_ = stamps_tail_;
    stamp->prev_ = nullptr;
    if (stamps_tail_)
      stamps_tail_->prev_ = stamp;
    else
      stamps_head_ = stamp;
    stamps_tail_ = stamp;
    ++stamps_size_;
  }

  void stamps_pop() {
    assert (stamps_size_);
    stamps_head_ = stamps_head_->prev_;
    if (stamps_head_)
      stamps_head_->next_ = nullptr;
    else
      stamps_tail_ = nullptr;
    --stamps_size_;
  }

  void renderQuad(const primitive_t& primitive, 
                  uint32_t  x, 
                  uint32_t  y, 
                  fixed16_t e0, 
                  fixed16_t e1, 
                  fixed16_t e2) {
    uint32_t mask = 0;
    std::array<vec3_fx_t, 4> bcoords = {vec3_fx_t{fxZero, fxZero, fxZero}, 
                                        vec3_fx_t{fxZero, fxZero, fxZero}, 
                                        vec3_fx_t{fxZero, fxZero, fxZero}, 
                                        vec3_fx_t{fxZero, fxZero, fxZero}};

    for (uint32_t j = 0; j < 2; ++j) {
      auto ee0 = e0;
      auto ee1 = e1;
      auto ee2 = e2;
      for (uint32_t i = 0; i < 2; ++i) {
        // test if pixel overlaps triangle
        if (ee0 >= fxZero && ee1 >= fxZero && ee2 >= fxZero) {
          // test if the pixel overlaps rendering region
          if ((x+i) < dst_width_ && (y+j) < dst_height_) {
            uint32_t f = j * 2 + i;          
            mask |= (1 << f);                
            bcoords[f].x = ee0;
            bcoords[f].y = ee1;
            bcoords[f].z = ee2;          
          }
        }
        // update edge equation x components
        ee0 += primitive.edges[0].x;
        ee1 += primitive.edges[1].x;
        ee2 += primitive.edges[2].x;
      }
      // update edge equation y components
      e0 += primitive.edges[0].y;
      e1 += primitive.edges[1].y;
      e2 += primitive.edges[2].y;
    }
    
    if (mask) {
      // add stamp to queue
      auto pos_x = x / 2;
      auto pos_y = y / 2;
      // printf("*** raster%d-quad: x_loc = %d, y_loc = %d, pid = %d, mask=%d, bcoords = %d %d %d %d, %d %d %d %d, %d %d %d %d\n",
      //   index_, pos_x, pos_y, pid_, mask,
      //   bcoords[0].x.data(), bcoords[1].x.data(), bcoords[2].x.data(), bcoords[3].x.data(),
      //   bcoords[0].y.data(), bcoords[1].y.data(), bcoords[2].y.data(), bcoords[3].y.data(),
      //   bcoords[0].z.data(), bcoords[1].z.data(), bcoords[2].z.data(), bcoords[3].z.data());
      this->stamps_push(new RasterUnit::Stamp(pos_x, pos_y, mask, bcoords, pid_));
    }
  }

  void renderBlock(uint32_t subBlockLogSize, 
                   const primitive_t& primitive, 
                   uint32_t  x, 
                   uint32_t  y, 
                   fixed16_t e0, 
                   fixed16_t e1, 
                   fixed16_t e2) {
    // check if block overlap triangle    
    if ((e0 + (primitive.extents[0] << subBlockLogSize)) < fxZero 
     || (e1 + (primitive.extents[1] << subBlockLogSize)) < fxZero
     || (e2 + (primitive.extents[2] << subBlockLogSize)) < fxZero) {
      /*printf("*** raster%d-block: pass=0, level=%d, x=%d, y=%d, edge_eval={0x%x, 0x%x, 0x%x}, extend={0x%x, 0x%x, 0x%x}\n", 
          index_, (tile_logsize_ - subBlockLogSize), x, y, e0, e1, e2, 
          (primitive.extents[0] << subBlockLogSize),
          (primitive.extents[1] << subBlockLogSize),
          (primitive.extents[2] << subBlockLogSize));*/
      return; 
     } else {
       /*printf("*** raster%d-block: pass=1, level=%d, x=%d, y=%d, edge_eval={0x%x, 0x%x, 0x%x}, extend={0x%x, 0x%x, 0x%x}\n", 
          index_, (tile_logsize_ - subBlockLogSize), x, y, e0, e1, e2, 
          (primitive.extents[0] << subBlockLogSize),
          (primitive.extents[1] << subBlockLogSize),
          (primitive.extents[2] << subBlockLogSize));*/
     }
  
    if (subBlockLogSize > 1) {
      // printf("*** raster%d-block: x=%d, y=%d\n", index_, x, y);

      --subBlockLogSize;
      auto subBlockSize = 1 << subBlockLogSize;
      // draw top-left subtile
      {
        auto sx  = x;
        auto sy  = y;
        auto se0 = e0;
        auto se1 = e1;
        auto se2 = e2;
        this->renderBlock(subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw top-right subtile
      {
        auto sx  = x + subBlockSize;
        auto sy  = y;
        auto se0 = e0 + (primitive.edges[0].x << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subBlockLogSize);
        this->renderBlock(subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-left subtile
      {
        auto sx  = x;
        auto sy  = y + subBlockSize;
        auto se0 = e0 + (primitive.edges[0].y << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].y << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].y << subBlockLogSize);
        this->renderBlock(subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-right subtile
      {
        auto sx  = x + subBlockSize;
        auto sy  = y + subBlockSize;
        auto se0 = e0 + (primitive.edges[0].x << subBlockLogSize) + (primitive.edges[0].y << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subBlockLogSize) + (primitive.edges[1].y << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subBlockLogSize) + (primitive.edges[2].y << subBlockLogSize);
        this->renderBlock(subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }
    } else {
      // draw low-level block
      this->renderQuad(primitive, x, y, e0, e1, e2);
    }
  }

  void renderTile(uint32_t subTileLogSize, 
                  const primitive_t& primitive, 
                  uint32_t  x, 
                  uint32_t  y, 
                  fixed16_t e0, 
                  fixed16_t e1, 
                  fixed16_t e2) {
    // check if tile overlap triangle    
    if ((e0 + (primitive.extents[0] << subTileLogSize)) < fxZero 
     || (e1 + (primitive.extents[1] << subTileLogSize)) < fxZero
     || (e2 + (primitive.extents[2] << subTileLogSize)) < fxZero) {
      /*printf("*** raster%d-tile: pass=0, level=%d, x=%d, y=%d, edge_eval={0x%x, 0x%x, 0x%x}, extend={0x%x, 0x%x, 0x%x}\n", 
        index_, (tile_logsize_ - subTileLogSize), x, y, e0, e1, e2, 
        (primitive.extents[0] << subTileLogSize),
        (primitive.extents[1] << subTileLogSize),
        (primitive.extents[2] << subTileLogSize));*/
      return; 
    } else {
      /*printf("*** raster%d-tile: pass=1, level=%d, x=%d, y=%d, edge_eval={0x%x, 0x%x, 0x%x}, extend={0x%x, 0x%x, 0x%x}\n", 
        index_, (tile_logsize_ - subTileLogSize), x, y, e0, e1, e2, 
        (primitive.extents[0] << subTileLogSize),
        (primitive.extents[1] << subTileLogSize),
        (primitive.extents[2] << subTileLogSize));*/
    }
    
    if (subTileLogSize > block_logsize_) {
      // printf("*** raster%d-tile: x=%d, y=%d\n", index_, x, y);

      --subTileLogSize;
      auto subTileSize = 1 << subTileLogSize;
      // draw top-left subtile
      {
        auto sx  = x;
        auto sy  = y;
        auto se0 = e0;
        auto se1 = e1;
        auto se2 = e2;
        this->renderTile(subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw top-right subtile
      {
        auto sx  = x + subTileSize;
        auto sy  = y;
        auto se0 = e0 + (primitive.edges[0].x << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subTileLogSize);
        this->renderTile(subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-left subtile
      {
        auto sx  = x;
        auto sy  = y + subTileSize;
        auto se0 = e0 + (primitive.edges[0].y << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].y << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].y << subTileLogSize);
        this->renderTile(subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-right subtile
      {
        auto sx  = x + subTileSize;
        auto sy  = y + subTileSize;
        auto se0 = e0 + (primitive.edges[0].x << subTileLogSize) + (primitive.edges[0].y << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subTileLogSize) + (primitive.edges[1].y << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subTileLogSize) + (primitive.edges[2].y << subTileLogSize);
        this->renderTile(subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }
    } else {
      if (block_logsize_ > 1) {
        // draw low-level block
        this->renderBlock(subTileLogSize, primitive, x, y, e0, e1, e2);
      } else {
        // draw low-level quad
        this->renderQuad(primitive, x, y, e0, e1, e2);
      }
    }
  }

  void renderNextPrimitive() {    
    if (0 == prim_count_) {
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
      //printf("*** raster%d-mem: add=%d, tile_x=%d, tile_y=%d\n", index_, tbuf_addr_, tile_x_, tile_y_);
      tbuf_addr_ += 4;
      
      mem_->read(&prim_header, tbuf_addr_, 4);
      mem_trace.header_addrs.push_back(tbuf_addr_);
      prim_offset_ = (prim_header & 0xffff);
      prim_count_  = (prim_header >> 16);
      //printf("*** raster%d-mem: add=%d, prim_off=%d, prim_cnt=%d\n", index_, tbuf_addr_, prim_offset_, prim_count_);
      tbuf_addr_ += 4;

      assert(prim_count_ > 0);
      
      cur_prim_ = 0;
    }

    assert(!mem_traces_.empty());
    auto& mem_trace = mem_traces_.back();
    prim_mem_trace_t prim_trace;

    // read next primitive index from tile buffer
    prim_trace.prim_addr = tbuf_addr_  + prim_offset_;
    mem_->read(&pid_, prim_trace.prim_addr, 4);    
    //printf("*** raster%d-mem: add=%d, pid=%d\n", index_, prim_trace.prim_addr, pid_);
    prim_offset_ += 4;

    uint32_t x = tile_x_;
    uint32_t y = tile_y_;

    //printf("*** raster%d-primitive: tile=%d/%d, prim=%d/%d, pid=%d, tx=%d, ty=%d\n", index_, cur_tile_, num_tiles_, cur_prim_, prim_count_, pid_, x, y);

    // get primitive edges
    primitive_t primitive;
    auto pbuf_addr = pbuf_baseaddr_ + pid_ * pbuf_stride_;
    for (int i = 0; i < 3; ++i) {
      mem_->read(&primitive.edges[i].x, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.x=%d\n", index_, pbuf_addr, primitive.edges[i].x.data());      
      pbuf_addr += 4;
      
      mem_->read(&primitive.edges[i].y, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.y=%d\n", index_, pbuf_addr, primitive.edges[i].y.data());
      pbuf_addr += 4;
      
      mem_->read(&primitive.edges[i].z, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster%d-mem: add=%d, edge.z=%d\n", index_, pbuf_addr, primitive.edges[i].z.data());
      pbuf_addr += 4;      
    }

    /*printf("*** raster%d-edges={{0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}}\n", 
      index_, 
      primitive.edges[0].x.data(), primitive.edges[0].y.data(), primitive.edges[0].z.data(),
      primitive.edges[1].x.data(), primitive.edges[1].y.data(), primitive.edges[1].z.data(),
      primitive.edges[2].x.data(), primitive.edges[2].y.data(), primitive.edges[2].z.data());*/

    // Add tile corner edge offsets
    primitive.extents[0] = calcEdgeExtents(primitive.edges[0]);
    primitive.extents[1] = calcEdgeExtents(primitive.edges[1]);
    primitive.extents[2] = calcEdgeExtents(primitive.edges[2]);

    // Evaluate edge equation for the starting tile
    auto e0 = evalEdgeFunction(primitive.edges[0], x, y);
    auto e1 = evalEdgeFunction(primitive.edges[1], x, y);
    auto e2 = evalEdgeFunction(primitive.edges[2], x, y);

    // Render the tile
    if (tile_logsize_ > block_logsize_) {
      this->renderTile(tile_logsize_, primitive, x, y, e0, e1, e2);
    } else {
      this->renderBlock(block_logsize_, primitive, x, y, e0, e1, e2);
    }

    // printf("*** raster%d: generated %d stamps\n", index_, stamps_size_);
    prim_trace.stamps = stamps_size_;
    mem_trace.primitives.push_back(prim_trace);

    // Move to next primitive
    ++cur_prim_;
    if (cur_prim_ == prim_count_) {
      mem_trace.end_of_tile = true;
      // Move to next tile      
      cur_tile_  += NUM_CLUSTERS;
      tbuf_addr_ += (NUM_CLUSTERS-1) * 8;
      prim_count_ = 0;
    }
  }

public:
  Rasterizer(uint32_t index,
             const Arch& arch,
             const RasterUnit::DCRS& dcrs,               
             uint32_t tile_logsize, 
             uint32_t block_logsize) 
    : index_(index)
    , arch_(arch)
    , dcrs_(dcrs)    
    , tile_logsize_(tile_logsize)
    , block_logsize_(block_logsize)
    , stamps_head_(nullptr)
    , stamps_tail_(nullptr)
    , stamps_size_(0)
    , done_(false) {
    assert(block_logsize >= 1);
    assert(tile_logsize >= block_logsize);
  }

  ~Rasterizer() {
    //--
  } 
  
  uint32_t id() const {
    return index_;
  }

  void initialize() {
    // get device configuration
    num_tiles_     = dcrs_.read(DCR_RASTER_TILE_COUNT);
    tbuf_baseaddr_ = dcrs_.read(DCR_RASTER_TBUF_ADDR);
    pbuf_baseaddr_ = dcrs_.read(DCR_RASTER_PBUF_ADDR);
    pbuf_stride_   = dcrs_.read(DCR_RASTER_PBUF_STRIDE);
    dst_width_     = dcrs_.read(DCR_RASTER_DST_SIZE) & 0xffff;
    dst_height_    = dcrs_.read(DCR_RASTER_DST_SIZE) >> 16;

    tbuf_addr_  = tbuf_baseaddr_ + index_ * 8;
    cur_tile_   = index_;
    cur_prim_   = 0;
    prim_count_ = 0;
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  RasterUnit::Stamp* fetch() {
    do {
      if (stamps_head_ == nullptr && cur_tile_ >= num_tiles_) {
        done_ = true;
        return nullptr;
      }
      if (stamps_head_ == nullptr) {
        this->renderNextPrimitive();
      }      
    } while (stamps_head_ == nullptr);
    auto stamp = stamps_head_;
    this->stamps_pop();
    return stamp;
  }
  
  bool done() const {
    return done_;
  }

  auto& mem_traces() {
    return mem_traces_;
  }

  auto& mem_traces() const {
    return mem_traces_;
  }
};

///////////////////////////////////////////////////////////////////////////////

class RasterUnit::Impl {
private:
  enum class e_mem_trace_state {
    header,
    primitive,
    edges
  };

  class CSR {
  private:
    RasterUnit::Stamp *stamp_;

  public:
    CSR() : stamp_(nullptr) { this->clear(); }    
    ~CSR() { this->clear(); }

    void clear() {
      if (stamp_) {
        delete stamp_;
        stamp_ = nullptr;
      }
    }

    void set_stamp(RasterUnit::Stamp *stamp) {
      if (stamp_)
        delete stamp_;
      stamp_ = stamp;
    }

    RasterUnit::Stamp* get_stamp() const {
      assert(stamp_);
      return stamp_;
    }
  };

  struct pending_req_t {
    uint32_t count;
  };

  RasterUnit* simobject_;        
  const Arch& arch_;
  Rasterizer rasterizer_;
  std::vector<CSR> csrs_;
  std::queue<uint32_t> stamps_;  
  uint32_t fetched_stamps_;
  HashTable<pending_req_t> pending_reqs_;
  e_mem_trace_state mem_trace_state_;
  PerfStats perf_stats_; 
  uint32_t cores_per_unit_;
  
public:
  Impl(RasterUnit* simobject,     
       uint32_t index,
       uint32_t cores_per_unit,
       const Arch &arch,
       const DCRS& dcrs,       
       const Config& config) 
    : simobject_(simobject)    
    , arch_(arch)
    , rasterizer_(index, arch, dcrs, config.tile_logsize, config.block_logsize)
    , csrs_(cores_per_unit * arch.num_warps() * arch.num_threads())
    , pending_reqs_(RASTER_MEM_QUEUE_SIZE)    
    , mem_trace_state_(e_mem_trace_state::header)
    , cores_per_unit_(cores_per_unit)
  {}

  ~Impl() {}

  uint32_t id() const {
    return rasterizer_.id();
  }

  void clear() {
    rasterizer_.initialize();
    mem_trace_state_ = e_mem_trace_state::header;
    for (auto& csr : csrs_) {
      csr.clear();
    }
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
      --entry.count; // track remaining blocks 
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

    perf_stats_.stalls += simobject_->Output.stalled();

    if (mem_traces.empty())
      return;

    // check pending queue is empty    
    if (!pending_reqs_.empty())          
      return;

    auto& mem_trace = mem_traces.front();

    std::vector<uint32_t> addresses;

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
      simobject_->MemReqs.send(mem_req, 1);
      ++perf_stats_.reads;
    }
  }

  void attach_ram(RAM* mem) {
    rasterizer_.attach_ram(mem);
  }
  
  uint32_t csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr) {
    uint32_t lcid = cid % cores_per_unit_;
    uint32_t index = (lcid * arch_.num_warps() + wid) * arch_.num_threads() + tid;
    auto& csr = csrs_.at(index);
    auto stamp = csr.get_stamp();

    uint32_t value;

    switch (addr) {
    case CSR_RASTER_POS_MASK:
      value = (stamp->y << (4 + RASTER_DIM_BITS-1)) | (stamp->x << 4) | stamp->mask;      
      DT(2, "raster-csr: cid=" << std::dec << cid << ", wid=" << wid <<", tid=" << tid << ", pos_mask=" << value);
      break;
    case CSR_RASTER_BCOORD_X0:
    case CSR_RASTER_BCOORD_X1:
    case CSR_RASTER_BCOORD_X2:
    case CSR_RASTER_BCOORD_X3:
      value = stamp->bcoords.at(addr - CSR_RASTER_BCOORD_X0).x.data();
      DT(2, "raster-csr: cid=" << std::dec << cid << ", wid=" << wid <<", tid=" << tid << ", bcoord.x" << (addr - CSR_RASTER_BCOORD_X0) << ", value=" << std::hex << value);
      break;
    case CSR_RASTER_BCOORD_Y0:
    case CSR_RASTER_BCOORD_Y1:
    case CSR_RASTER_BCOORD_Y2:
    case CSR_RASTER_BCOORD_Y3:
      value = stamp->bcoords.at(addr - CSR_RASTER_BCOORD_Y0).y.data();
      DT(2, "raster-csr: cid=" << std::dec << cid << ", wid=" << wid <<", tid=" << tid << ", bcoord.y" << (addr - CSR_RASTER_BCOORD_Y0) << ", value=" << std::hex << value);
      break;
    case CSR_RASTER_BCOORD_Z0:
    case CSR_RASTER_BCOORD_Z1:
    case CSR_RASTER_BCOORD_Z2:
    case CSR_RASTER_BCOORD_Z3:
      value = stamp->bcoords.at(addr - CSR_RASTER_BCOORD_Z0).z.data();
      DT(2, "raster-csr: cid=" << std::dec << cid << ", wid=" << wid <<", tid=" << tid << ", bcoord.z" << (addr - CSR_RASTER_BCOORD_Z0) << ", value=" << std::hex << value);
      break;
    default:
      std::abort();
    }
    
    return value;
  }

  void csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
    __unused (cid);
    __unused (wid);
    __unused (tid);
    __unused (addr);
    __unused (value);
  } 

  uint32_t fetch(uint32_t cid, uint32_t wid, uint32_t tid) {
    auto stamp = rasterizer_.fetch();
    if (nullptr == stamp)
      return 0;

    uint32_t lcid = cid % cores_per_unit_;
    uint32_t index = (lcid * arch_.num_warps() + wid) * arch_.num_threads() + tid;
    auto& csr = csrs_.at(index);
    csr.set_stamp(stamp);
    
    DT(2, "raster-fetch: cid=" << std::dec << cid << ", wid=" << wid <<", tid=" << tid << ", x=" << stamp->x << ", y=" << stamp->y << ", mask=" << stamp->mask << ", pid=" << stamp->pid << ", bcoords=" << std::hex
      <<  "{{0x" << stamp->bcoords[0].x.data() << ", 0x" << stamp->bcoords[0].y.data() << ", 0x" << stamp->bcoords[0].z.data() << "}"
      << ", {0x" << stamp->bcoords[1].x.data() << ", 0x" << stamp->bcoords[1].y.data() << ", 0x" << stamp->bcoords[1].z.data() << "}"
      << ", {0x" << stamp->bcoords[2].x.data() << ", 0x" << stamp->bcoords[2].y.data() << ", 0x" << stamp->bcoords[2].z.data() << "}"
      << ", {0x" << stamp->bcoords[3].x.data() << ", 0x" << stamp->bcoords[3].y.data() << ", 0x" << stamp->bcoords[3].z.data() << "}}");

    return (stamp->pid << 1) | 1;
  }

  const PerfStats& perf_stats() const { 
    return perf_stats_; 
  }
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
  impl_->clear();
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

uint32_t RasterUnit::csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(cid, wid, tid, addr);
}

void RasterUnit::csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(cid, wid, tid, addr, value);
}

uint32_t RasterUnit::fetch(uint32_t cid, uint32_t wid, uint32_t tid) {
  return impl_->fetch(cid, wid, tid);
}

const RasterUnit::PerfStats& RasterUnit::perf_stats() const {
  return impl_->perf_stats();
}