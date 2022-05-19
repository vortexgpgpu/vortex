#include "raster_unit.h"
#include "core.h"
#include <VX_config.h>
#include "mempool.h"
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

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
  uint32_t num_prims_;    
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
    std::array<vec3_fx_t, 4> bcoords;

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
      // printf("raster-quad: x_loc = %d, y_loc = %d, pid = %d, mask=%d, bcoords = %d %d %d %d, %d %d %d %d, %d %d %d %d\n",
      //   pos_x, pos_y, pid_, mask,
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
     || (e2 + (primitive.extents[2] << subBlockLogSize)) < fxZero)
      return; 
  
    if (subBlockLogSize > 1) {
      // printf("*** raster-block: x=%d, y=%d\n", x, y);

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
     || (e2 + (primitive.extents[2] << subTileLogSize)) < fxZero)
      return; 
    
    if (subTileLogSize > block_logsize_) {
      // printf("*** raster-tile: x=%d, y=%d\n", x, y);

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
    if (0 == num_prims_) {
      mem_traces_.push_back({});      
      auto& mem_trace = mem_traces_.back();
      mem_trace.end_of_tile = false;

      uint32_t tile_xy;

      // read next tile header from tile buffer
      mem_->read(&tile_xy, tbuf_addr_, 4);
      mem_trace.header_addrs.push_back(tbuf_addr_);
      // printf("*** raster-mem: add=%d, tile_xy=%d\n", tbuf_addr_, tile_xy);
      tbuf_addr_ += 4;
      
      mem_->read(&num_prims_, tbuf_addr_, 4);
      mem_trace.header_addrs.push_back(tbuf_addr_);
      // printf("*** raster-mem: add=%d, num_prims=%d\n", tbuf_addr_, num_prims_);
      tbuf_addr_ += 4;

      assert(num_prims_ > 0);
      tile_x_ = (tile_xy & 0xffff) << tile_logsize_;
      tile_y_ = (tile_xy >> 16) << tile_logsize_;
      cur_prim_ = 0;
    }

    assert(!mem_traces_.empty());
    auto& mem_trace = mem_traces_.back();
    prim_mem_trace_t prim_trace;

    // read next primitive index from tile buffer
    mem_->read(&pid_, tbuf_addr_, 4);
    prim_trace.prim_addr = tbuf_addr_;
    // printf("*** raster-mem: add=%d, pid=%d\n", tbuf_addr_, pid_);
    tbuf_addr_ += 4;

    uint32_t x = tile_x_;
    uint32_t y = tile_y_;

    // printf("*** raster-primitive: tile=%d/%d, prim=%d/%d, pid=%d, tx=%d, ty=%d\n", cur_tile_, num_tiles_, cur_prim_, num_prims_, pid_, x, y);

    // get primitive edges
    primitive_t primitive;
    auto pbuf_addr = pbuf_baseaddr_ + pid_ * pbuf_stride_;
    for (int i = 0; i < 3; ++i) {
      mem_->read(&primitive.edges[i].x, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster-mem: add=%d, edge.x=%d\n", pbuf_addr, primitive.edges[i].x.data());      
      pbuf_addr += 4;
      
      mem_->read(&primitive.edges[i].y, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster-mem: add=%d, edge.y=%d\n", pbuf_addr, primitive.edges[i].y.data());
      pbuf_addr += 4;
      
      mem_->read(&primitive.edges[i].z, pbuf_addr, 4);
      prim_trace.edge_addrs.push_back(pbuf_addr);
      // printf("*** raster-mem: add=%d, edge.z=%d\n", pbuf_addr, primitive.edges[i].z.data());
      pbuf_addr += 4;      
    }

    // printf("*** raster-edge0: a=%d, b=%d, c=%d\n", primitive.edges[0].x.data(), primitive.edges[0].y.data(), primitive.edges[0].z.data());
    // printf("*** raster-edge1: a=%d, b=%d, c=%d\n", primitive.edges[1].x.data(), primitive.edges[1].y.data(), primitive.edges[1].z.data());
    // printf("*** raster-edge2: a=%d, b=%d, c=%d\n", primitive.edges[2].x.data(), primitive.edges[2].y.data(), primitive.edges[2].z.data());

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

    // printf("*** raster: generated %d stamps\n", stamps_size_);
    prim_trace.stamps = stamps_size_;
    mem_trace.primitives.push_back(prim_trace);

    // Move to next primitive
    ++cur_prim_;
    if (cur_prim_ == num_prims_) {
      mem_trace.end_of_tile = true;
      // Move to next tile      
      ++cur_tile_;
      num_prims_ = 0;
    }
  }

public:
  Rasterizer(const Arch& arch,
              const RasterUnit::DCRS& dcrs, 
              uint32_t tile_logsize, 
              uint32_t block_logsize) 
    : arch_(arch)
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

  void initialize() {
    // get device configuration
    num_tiles_     = dcrs_.read(DCR_RASTER_TILE_COUNT);
    tbuf_baseaddr_ = dcrs_.read(DCR_RASTER_TBUF_ADDR);
    pbuf_baseaddr_ = dcrs_.read(DCR_RASTER_PBUF_ADDR);
    pbuf_stride_   = dcrs_.read(DCR_RASTER_PBUF_STRIDE);
    dst_width_     = dcrs_.read(DCR_RASTER_DST_SIZE) & 0xffff;
    dst_height_    = dcrs_.read(DCR_RASTER_DST_SIZE) >> 16;

    tbuf_addr_ = tbuf_baseaddr_;
    cur_tile_  = 0;
    cur_prim_  = 0;
    num_prims_ = 0;
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  RasterUnit::Stamp* fetch() {
    do {
      if (stamps_head_ == nullptr && cur_tile_ == num_tiles_) {
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

class RasterUnit::Impl {
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
  Rasterizer rasterizer_;
  PerfStats perf_stats_;  
  HashTable<pending_req_t> pending_reqs_;
  e_mem_trace_state mem_trace_state_;

public:
  Impl(RasterUnit* simobject,     
       const Arch &arch,
       const DCRS& dcrs, 
       uint32_t tile_logsize, 
       uint32_t block_logsize) 
    : simobject_(simobject)
    , arch_(arch)
    , rasterizer_(arch, dcrs, tile_logsize, block_logsize)
    , pending_reqs_(RASTER_MEM_QUEUE_SIZE)
    , mem_trace_state_(e_mem_trace_state::header)
  {}

  ~Impl() {}

  void clear() {
    rasterizer_.initialize();
    mem_trace_state_ = e_mem_trace_state::header;
  }  

  void attach_ram(RAM* mem) {
    rasterizer_.attach_ram(mem);
  }

  bool done() const {
    return rasterizer_.done() 
        && rasterizer_.mem_traces().empty();
  }

  RasterUnit::Stamp* fetch() {      
    return rasterizer_.fetch();
  }

  void tick() {
    auto& mem_traces = rasterizer_.mem_traces();

    // handle memory response
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

          StampRsp stampRsp{primitive.stamps};
          simobject_->Output.send(stampRsp, 8);
          
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
      mem_req.addr    = addr;
      mem_req.write   = false;
      mem_req.tag     = tag;
      mem_req.core_id = 0;
      mem_req.uuid    = 0;
      simobject_->MemReqs.send(mem_req, 1);
      ++perf_stats_.reads;
    }
  }

  const PerfStats& perf_stats() const { 
    return perf_stats_; 
  }
};

///////////////////////////////////////////////////////////////////////////////

RasterUnit::RasterUnit(const SimContext& ctx, 
                       const char* name,                        
                       const Arch &arch, 
                       const DCRS& dcrs,
                       uint32_t tile_logsize, 
                       uint32_t block_logsize) 
  : SimObject<RasterUnit>(ctx, name)
  , MemReqs(this)
  , MemRsps(this)
  , Output(this)
  , impl_(new Impl(this, arch, dcrs, tile_logsize, block_logsize)) 
{}

RasterUnit::~RasterUnit() {
  delete impl_;
}

void RasterUnit::reset() {
  impl_->clear();
}

void RasterUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

bool RasterUnit::done() const {
  return impl_->done();
}

RasterUnit::Stamp* RasterUnit::fetch() {
  return impl_->fetch();
}

void RasterUnit::tick() {
  impl_->tick();
}

const RasterUnit::PerfStats& RasterUnit::perf_stats() const {
  return impl_->perf_stats();
}