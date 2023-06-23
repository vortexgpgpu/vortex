#pragma once

#include "common.h"
#include <vx_intrinsics.h>

using fixed16_t = cocogfx::TFixed<16>;
using vec2_fx_t = cocogfx::TVector2<fixed16_t>;
using vec3_fx_t = cocogfx::TVector3<fixed16_t>;

void shader_function_sw_rast_cb(uint32_t  pos_mask,
                                graphics::vec3e_t bcoords[4],
                                uint32_t  pid,
                                void* arg);

class Rasterizer : graphics::Rasterizer {
public:

  Rasterizer() : graphics::Rasterizer(
    shader_function_sw_rast_cb, 
    nullptr,
    RASTER_TILE_LOGSIZE,
    RASTER_BLOCK_LOGSIZE
  ) {}

  void configure(const graphics::RasterDCRS& dcrs, uint32_t log_num_tasks) {    
    graphics::Rasterizer::configure(dcrs);    
    num_tiles_     = dcrs.read(DCR_RASTER_TILE_COUNT);
    tbuf_baseaddr_ = uint64_t(dcrs.read(DCR_RASTER_TBUF_ADDR)) << 6;
    pbuf_baseaddr_ = uint64_t(dcrs.read(DCR_RASTER_PBUF_ADDR)) << 6;
    pbuf_stride_   = dcrs.read(DCR_RASTER_PBUF_STRIDE);
    log_num_tasks_ = log_num_tasks;
  }

  void render(uint32_t task_id) const {  
    uint32_t num_tasks = 1 << log_num_tasks_;
    auto tile_buffer = reinterpret_cast<const graphics::rast_tile_header_t*>(tbuf_baseaddr_);

    for (uint32_t cur_tile = task_id; cur_tile < num_tiles_; cur_tile += num_tasks) {
      auto tile_header = tile_buffer + cur_tile;

      uint32_t x = tile_header->tile_x << tile_logsize_;
      uint32_t y = tile_header->tile_y << tile_logsize_;
      uint32_t pids_count = tile_header->pids_count;
      auto pid_buf = reinterpret_cast<const uint32_t*>(tile_header + 1) + tile_header->pids_offset;

      while (pids_count--) {
        // read next primitive index from tile buffer
        uint32_t pid = *pid_buf++;
        
        // get primitive edges
        auto prim_buf = reinterpret_cast<graphics::FloatE*>(pbuf_baseaddr_ + pid * pbuf_stride_);
        graphics::vec3e_t edges[3] = {
          {prim_buf[0], prim_buf[1], prim_buf[2]},
          {prim_buf[3], prim_buf[4], prim_buf[5]},
          {prim_buf[6], prim_buf[7], prim_buf[8]}
        };

        // Render the primitive
        this->renderPrimitive(x, y, pid, edges);
      }
    }
  }

private:

  uint32_t num_tiles_;
  uint64_t tbuf_baseaddr_;    
  uint64_t pbuf_baseaddr_;
  uint32_t pbuf_stride_;
  uint32_t log_num_tasks_;
};

///////////////////////////////////////////////////////////////////////////////

class RenderOutput {
public:

  void configure(const graphics::RopDCRS& dcrs) {
    depthStencil_.configure(dcrs);
    blender_.configure(dcrs);

    // get device configuration
    zbuf_baseaddr_ = uint64_t(dcrs.read(DCR_ROP_ZBUF_ADDR)) << 6;
    zbuf_pitch_    = dcrs.read(DCR_ROP_ZBUF_PITCH);
    depth_writemask_ = dcrs.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;
    stencil_front_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) & 0xffff;
    stencil_back_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) >> 16;

    cbuf_baseaddr_ = uint64_t(dcrs.read(DCR_ROP_CBUF_ADDR)) << 6;
    cbuf_pitch_    = dcrs.read(DCR_ROP_CBUF_PITCH);
    auto cbuf_writemask = dcrs.read(DCR_ROP_CBUF_WRITEMASK) & 0xf;
    cbuf_writemask_ = (((cbuf_writemask >> 0) & 0x1) * 0x000000ff) 
                    | (((cbuf_writemask >> 1) & 0x1) * 0x0000ff00) 
                    | (((cbuf_writemask >> 2) & 0x1) * 0x00ff0000) 
                    | (((cbuf_writemask >> 3) & 0x1) * 0xff000000);
    color_read_  = (cbuf_writemask != 0xf);
    color_write_ = (cbuf_writemask != 0x0);
  }

  void write(unsigned x, 
             unsigned y, 
             unsigned is_backface, 
             unsigned color, 
             unsigned depth) const {
    auto blend_enabled = blender_.enabled();
    auto depth_enabled = depthStencil_.depth_enabled();
    auto stencil_enabled = depthStencil_.stencil_enabled(is_backface);    

    uint32_t depthstencil;    
    uint32_t dst_depthstencil;
    uint32_t dst_color;    

    this->read(depth_enabled, stencil_enabled, blend_enabled, x, y, &dst_depthstencil, &dst_color);
    
    auto ds_passed = !(depth_enabled || stencil_enabled)
                  || depthStencil_.test(is_backface, depth, dst_depthstencil, &depthstencil);
    
    if (blend_enabled && ds_passed) {
      color = blender_.blend(color, dst_color);
    }
    
    this->write(depth_enabled, stencil_enabled, ds_passed, is_backface, dst_depthstencil, dst_color, x, y, depthstencil, color);
  }

private:

  void read(bool depth_enable,
            bool stencil_enable, 
            bool blend_enable,
            uint32_t x, 
            uint32_t y,
            uint32_t* depthstencil,
            uint32_t* color) const {
    if (depth_enable || stencil_enable) {
      uint64_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;
      *depthstencil = *reinterpret_cast<const uint32_t*>(zbuf_addr);
    }

    if (color_write_ && (color_read_ || blend_enable)) {
      uint64_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      *color = *reinterpret_cast<const uint32_t*>(cbuf_addr);
    }
  }

  void write(bool depth_enable,
             bool stencil_enable, 
             bool ds_passed,
             bool is_backface,
             uint32_t dst_depthstencil,
             uint32_t dst_color,
             uint32_t x, 
             uint32_t y, 
             uint32_t depthstencil, 
             uint32_t color) const {
    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    auto ds_writeMask = ((depth_enable && ds_passed && depth_writemask_) ? ROP_DEPTH_MASK : 0) 
                      | (stencil_enable ? (stencil_writemask << ROP_DEPTH_BITS) : 0);
    if (ds_writeMask != 0) {      
      uint32_t write_value = (dst_depthstencil & ~ds_writeMask) | (depthstencil & ds_writeMask);
      uint64_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;        
      *reinterpret_cast<uint32_t*>(zbuf_addr) = write_value;
    }
    if (color_write_ && ds_passed) {   
      uint32_t write_value = (dst_color & ~cbuf_writemask_) | (color & cbuf_writemask_);
      uint64_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      *reinterpret_cast<uint32_t*>(cbuf_addr) = write_value;
    }
  }

  graphics::DepthTencil depthStencil_;
  graphics::Blender     blender_;

  uint64_t zbuf_baseaddr_;
  uint32_t zbuf_pitch_;
  bool     depth_writemask_;
  uint32_t stencil_front_writemask_; 
  uint32_t stencil_back_writemask_;

  uint64_t cbuf_baseaddr_;
  uint32_t cbuf_pitch_;
  uint32_t cbuf_writemask_;
  bool     color_read_;
  bool     color_write_;
};

///////////////////////////////////////////////////////////////////////////////

class TextureSampler : public graphics::TextureSampler {
public:
  TextureSampler() : graphics::TextureSampler(
    memory_cb,
    nullptr
  ) {}
  
  ~TextureSampler() {}

private:
  static void memory_cb(uint32_t* out,
                        const uint64_t* addr,
                        uint32_t stride,
                        uint32_t size,
                        void* /*cb_arg*/) {
    switch (stride) {
    case 4:
      for (uint32_t i = 0; i < size; ++i) {
        out[i] = *reinterpret_cast<const uint32_t*>(addr[i]);
      }    
      break;
    case 2:
      for (uint32_t i = 0; i < size; ++i) {
        out[i] = *reinterpret_cast<const uint16_t*>(addr[i]);
      }    
      break;
    case 1:
      for (uint32_t i = 0; i < size; ++i) {
        out[i] = *reinterpret_cast<const uint8_t*>(addr[i]);
      }    
      break;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

class GpuSW {
public:

  void configure(kernel_arg_t* __UNIFORM__ kernel_arg) {
    rasterizer_.configure(kernel_arg->raster_dcrs, kernel_arg->log_num_tasks);
    renderOutput_.configure(kernel_arg->rop_dcrs);
    sampler_.configure(kernel_arg->tex_dcrs);
    kernel_arg_ = kernel_arg;
  }

  kernel_arg_t* kernel_arg() {
    return kernel_arg_;
  }

  void render(unsigned task_id) const {
    rasterizer_.render(task_id);
  }

  void rop(unsigned x, unsigned y, unsigned is_backface, unsigned color, unsigned depth) const {
    renderOutput_.write(x, y, is_backface, color, depth);
  }

  uint32_t tex(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const {
    return sampler_.read(stage, u, v, lod);
  }

private:  
  Rasterizer     rasterizer_;
  RenderOutput   renderOutput_;
  TextureSampler sampler_;
  kernel_arg_t*  kernel_arg_;
};
