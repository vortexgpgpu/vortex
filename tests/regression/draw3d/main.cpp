#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <array>
#include <assert.h>
#include <vortex.h>
#include "common.h"
#include "utils.h"
#include "model_quad.h"

using namespace cocogfx;

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
const char* input_file  = "soccer.png";
const char* output_file = "output.png";
const char* reference_file  = nullptr;
uint32_t clear_color = 0x00000000;
int tex_format = TEX_FORMAT_A8R8G8B8;
ePixelFormat tex_eformat = FORMAT_A8R8G8B8;
int tex_wrap = TEX_WRAP_CLAMP;
int tex_filter  = TEX_FILTER_POINT;
uint32_t dst_width  = 64;
uint32_t dst_height = 64;
const model_t& model = model_quad;

vx_device_h device = nullptr;
vx_buffer_h staging_buf = nullptr;
uint64_t tilebuf_addr;
uint64_t primbuf_addr;
uint64_t tbuf_addr;
uint64_t zbuf_addr;
uint64_t cbuf_addr;
kernel_arg_t kernel_arg;

static void show_usage() {
   std::cout << "Vortex 3D Rendering Test." << std::endl;
   std::cout << "Usage: [-i texture] [-o output] [-r reference] [-w width] [-h height]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:o:r:w:h:t:f:g:?")) != -1) {
    switch (c) {
    case 'i':
      input_file = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'r':
      reference_file = optarg;
      break;
    case 'w':
      dst_width = std::atoi(optarg);
      break;
    case 'h':
      dst_height = std::atoi(optarg);
      break;
    case 'f':
      tex_format  = std::atoi(optarg);
      switch (tex_format) {
      case TEX_FORMAT_A8R8G8B8: tex_eformat = FORMAT_A8R8G8B8; break;
      case TEX_FORMAT_R5G6B5: tex_eformat = FORMAT_R5G6B5; break;
      case TEX_FORMAT_A1R5G5B5: tex_eformat = FORMAT_A1R5G5B5; break;
      case TEX_FORMAT_A4R4G4B4: tex_eformat = FORMAT_A4R4G4B4; break;
      case TEX_FORMAT_A8L8: tex_eformat = FORMAT_A8L8; break;
      case TEX_FORMAT_L8: tex_eformat = FORMAT_L8; break;
      case TEX_FORMAT_A8: tex_eformat = FORMAT_A8; break;
      default:
        std::cout << "Error: invalid texture format: " << tex_format << std::endl;
        exit(1);
      }
      break;
    case 'g':
      tex_filter = std::atoi(optarg);
      break;
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (staging_buf) {
    vx_buf_free(staging_buf);
  }
  if (device) {
    vx_mem_free(device, tilebuf_addr);
    vx_mem_free(device, primbuf_addr);
    vx_mem_free(device, tbuf_addr);    
    vx_mem_free(device, zbuf_addr);
    vx_mem_free(device, cbuf_addr);
    vx_dev_close(device);
  }
}

int render(uint32_t buf_addr, uint32_t buf_size, uint32_t width, uint32_t height) {
  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));
  
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(staging_buf, buf_addr, buf_size, 0));

  std::vector<uint8_t> dst_pixels(buf_size);
  auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
  memcpy(dst_pixels.data(), buf_ptr, buf_size);

  // save output image
  std::cout << "save output image" << std::endl;
  //dump_image(dst_pixels, width, height, 4);
  RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, dst_pixels, width, height));

  return 0;
}

int main(int argc, char *argv[]) {    
  std::vector<uint8_t> tilebuf;
  std::vector<uint8_t> primbuf;

  std::vector<uint8_t> texbuf;    
  std::vector<uint32_t> mip_offsets;
  uint32_t tex_width;
  uint32_t tex_height;
  
  // parse command arguments
  parse_args(argc, argv);

  uint32_t tile_size = 1 << RASTER_TILE_LOGSIZE;

  if (!ispow2(dst_width)) {
    std::cout << "Error: only power of two dst_width supported: dst_width=" << dst_width << std::endl;
    return -1;
  }

  if (!ispow2(dst_height)) {
    std::cout << "Error: only power of two dst_height supported: dst_height=" << dst_height << std::endl;
    return -1;
  }

  if (0 != (dst_width % tile_size)) {
    std::cout << "Error: dst_with must be divisible by tile_size" << std::endl;
    return -1;
  }

  if (0 != (dst_height % tile_size)) {
    std::cout << "Error: dst_height must be divisible by tile_size" << std::endl;
    return -1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (0 == (isa_flags & (VX_ISA_EXT_RASTER | VX_ISA_EXT_ROP))) {
    std::cout << "raster or rop extensions not supported!" << std::endl;
    return -1;
  }

  {
    std::vector<uint8_t> staging;  
    RT_CHECK(LoadImage(input_file, tex_eformat, staging, &tex_width, &tex_height));    
    // check power of two support
    if (!ispow2(tex_width) || !ispow2(tex_height)) {
      std::cout << "Error: only power of two textures supported: width=" << tex_width << ", heigth=" << tex_height << std::endl;
      return -1;
    }
    uint32_t tex_bpp = Format::GetInfo(tex_eformat).BytePerPixel;
    uint32_t tex_pitch = tex_width * tex_bpp;
    RT_CHECK(GenerateMipmaps(texbuf, mip_offsets, staging, tex_eformat, tex_width, tex_height, tex_pitch));    
  }

  uint32_t primbuf_stride = sizeof(rast_prim_t);

  uint32_t tex_logwidth  = log2ceil(tex_width);
  uint32_t tex_logheight = log2ceil(tex_height);

  uint32_t zbuf_stride = 4;
  uint32_t zbuf_pitch = dst_width * zbuf_stride;
  uint32_t zbuf_size = dst_height * zbuf_pitch;

  uint32_t cbuf_stride = 4;
  uint32_t cbuf_pitch = dst_width * cbuf_stride;
  uint32_t cbuf_size = dst_width * cbuf_pitch;

  // Perform tile binning
  auto num_tiles = Binning(tilebuf, primbuf, model, dst_width, dst_height, tile_size);
  
  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, tilebuf.size(), &tilebuf_addr));
  RT_CHECK(vx_mem_alloc(device, primbuf.size(), &primbuf_addr));
  RT_CHECK(vx_mem_alloc(device, texbuf.size(), &tbuf_addr));  
  RT_CHECK(vx_mem_alloc(device, zbuf_size, &zbuf_addr));
  RT_CHECK(vx_mem_alloc(device, cbuf_size, &cbuf_addr));

  std::cout << "tilebuf_addr=0x" << std::hex << tilebuf_addr << std::endl;
  std::cout << "primbuf_addr=0x" << std::hex << primbuf_addr << std::endl;
  std::cout << "tbuf_addr=0x" << std::hex << tbuf_addr << std::endl;
  std::cout << "zbuf_addr=0x" << std::hex << zbuf_addr << std::endl;
  std::cout << "cbuf_addr=0x" << std::hex << cbuf_addr << std::endl;

  // allocate staging shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>({
      sizeof(kernel_arg_t), (uint32_t)tilebuf.size(), (uint32_t)primbuf.size(), zbuf_size, cbuf_size
    });
  RT_CHECK(vx_buf_alloc(device, alloc_size, &staging_buf));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    kernel_arg.dst_width  = dst_width;
    kernel_arg.dst_height = dst_height;
    kernel_arg.dst_stride = cbuf_stride;
    kernel_arg.dst_pitch  = cbuf_pitch;    
    kernel_arg.dst_addr   = cbuf_addr;

    auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload tiles buffer
  std::cout << "upload tiles buffer" << std::endl;      
  {    
    auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, tilebuf.data(), tilebuf.size());
    RT_CHECK(vx_copy_to_dev(staging_buf, tilebuf_addr, tilebuf.size(), 0));
  }

  // upload primitives buffer
  std::cout << "upload primitives buffer" << std::endl;      
  {    
    auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, primbuf.data(), primbuf.size());
    RT_CHECK(vx_copy_to_dev(staging_buf, primbuf_addr, primbuf.size(), 0));
  }

  // clear depth buffer
  std::cout << "clear depth buffer" << std::endl;      
  {    
    auto buf_ptr = (uint32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < (cbuf_size/4); ++i) {
      buf_ptr[i] = clear_color;
    }    
    RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.dst_addr, cbuf_size, 0));  
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;      
  {    
    auto buf_ptr = (uint32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < (cbuf_size/4); ++i) {
      buf_ptr[i] = clear_color;
    }    
    RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.dst_addr, cbuf_size, 0));  
  }

  // configure texture units
	vx_dcr_write(device, DCR_TEX_STAGE,  0);
	vx_dcr_write(device, DCR_TEX_LOGDIM, (tex_logheight << 16) | tex_logwidth);	
	vx_dcr_write(device, DCR_TEX_FORMAT, tex_format);
	vx_dcr_write(device, DCR_TEX_WRAP,   (tex_wrap << 16) | tex_wrap);
	vx_dcr_write(device, DCR_TEX_FILTER, tex_filter);
	vx_dcr_write(device, DCR_TEX_ADDR,   tbuf_addr);
	for (uint32_t i = 0; i < mip_offsets.size(); ++i) {
    assert(i < TEX_LOD_MAX);
		vx_dcr_write(device, DCR_TEX_MIPOFF(i), mip_offsets.at(i));
	};

  // configure raster units
  vx_dcr_write(device, DCR_RASTER_TBUF_ADDR,   tilebuf_addr);
  vx_dcr_write(device, DCR_RASTER_TILE_COUNT,  num_tiles);
  vx_dcr_write(device, DCR_RASTER_PBUF_ADDR,   primbuf_addr);
  vx_dcr_write(device, DCR_RASTER_PBUF_STRIDE, primbuf_stride);

  // configure rop buffers
  vx_dcr_write(device, DCR_ROP_ZBUF_ADDR,  zbuf_addr);
  vx_dcr_write(device, DCR_ROP_ZBUF_PITCH, zbuf_pitch);
  vx_dcr_write(device, DCR_ROP_CBUF_ADDR,  cbuf_addr);
  vx_dcr_write(device, DCR_ROP_CBUF_PITCH, cbuf_pitch);
  vx_dcr_write(device, DCR_ROP_CBUF_MASK, 0xffffffff);

  // configure rop depth states
  vx_dcr_write(device, DCR_ROP_DEPTH_FUNC, ROP_DEPTH_FUNC_LESS);
  vx_dcr_write(device, DCR_ROP_DEPTH_MASK, 1);
  vx_dcr_write(device, DCR_ROP_STENCIL_FUNC, ROP_DEPTH_FUNC_ALWAYS);
  vx_dcr_write(device, DCR_ROP_STENCIL_ZPASS, ROP_STENCIL_OP_KEEP);
  vx_dcr_write(device, DCR_ROP_STENCIL_ZPASS, ROP_STENCIL_OP_KEEP);
  vx_dcr_write(device, DCR_ROP_STENCIL_FAIL, ROP_STENCIL_OP_KEEP);
  vx_dcr_write(device, DCR_ROP_STENCIL_MASK, 0xff);
  vx_dcr_write(device, DCR_ROP_STENCIL_REF, 0);

  // configure rop blend stats
  vx_dcr_write(device, DCR_ROP_BLEND_MODE, (ROP_BLEND_MODE_ADD << 16) | ROP_BLEND_MODE_ADD);
  vx_dcr_write(device, DCR_ROP_BLEND_SRC,  (ROP_BLEND_FUNC_ONE << 16) | ROP_BLEND_FUNC_SRC_A);
  vx_dcr_write(device, DCR_ROP_BLEND_DST,  (ROP_BLEND_FUNC_ZERO << 16) | ROP_BLEND_FUNC_ONE_MINUS_SRC_A);
  vx_dcr_write(device, DCR_ROP_LOGIC_OP,  ROP_LOGIC_OP_COPY);

  // run tests
  std::cout << "render" << std::endl;
  RT_CHECK(render(cbuf_addr, cbuf_size, dst_width, dst_height));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();  

  if (reference_file) {
    auto errors = CompareImages(output_file, reference_file, FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED!" << std::endl;
      return errors;
    }
  } 

  return 0;
}