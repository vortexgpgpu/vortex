#include "utils.h"
#include <assert.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cocogfx/include/tga.hpp>
#include <cocogfx/include/png.hpp>
#include <cocogfx/include/bmp.hpp>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"

using namespace cocogfx;

using fixed16_t  = TFixed<16>;

using vec2d_f_t  = TVector2<float>;
using vec3d_f_t  = TVector3<float>;
using vec2d_fx_t = TVector2<fixed16_t>;
using vec4d_f_t  = TVector4<float>;
using rect_f_t   = TRect<float>;

// Evaluate edge function
static float evalEdgeFunction(const vec3d_f_t& e, uint32_t x, uint32_t y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// Calculate the edge extents for tile corners
static float calcEdgeExtents(const vec3d_f_t& e) {
  vec2d_f_t corners[4] = {{0.0f, 0.0f},  // 00
                          {1.0f, 0.0f},  // 10
                          {0.0f, 1.0f},  // 01
                          {1.0f, 1.0f}}; // 11
  auto i = (e.y >= 0.0f) ? ((e.x >= 0.0f) ? 3 : 2) : (e.x >= 0.0f) ? 1 : 0;
  return corners[i].x * e.x + corners[i].y * e.y;
}

static bool EdgeEquation(vec3d_f_t edges[3], 
                         const vec4d_f_t& v0, 
                         const vec4d_f_t& v1, 
                         const vec4d_f_t& v2) {
  // Calculate edge equation matrix
  auto a0 = (v1.y * v2.w) - (v2.y * v1.w);
  auto a1 = (v2.y * v0.w) - (v0.y * v2.w);
  auto a2 = (v0.y * v1.w) - (v1.y * v0.w);

  auto b0 = (v2.x * v1.w) - (v1.x * v2.w);
  auto b1 = (v0.x * v2.w) - (v2.x * v0.w);
  auto b2 = (v1.x * v0.w) - (v0.x * v1.w);

  auto c0 = (v1.x * v2.y) - (v2.x * v1.y);
  auto c1 = (v2.x * v0.y) - (v0.x * v2.y);
  auto c2 = (v0.x * v1.y) - (v1.x * v0.y);

  edges[0] = {a0, b0, c0};
  edges[1] = {a1, b1, c1};
  edges[2] = {a2, b2, c2};

  /*printf("E0.x=%f, E0.y=%f, E0.z=%f, E1.x=%f, E1.y=%f, E1.z=%f, E2.x=%f, E2.y=%f, E2.z=%f\n", 
      edges[0].x, edges[0].y, edges[0].z,
      edges[1].x, edges[1].y, edges[1].z,
      edges[2].x, edges[2].y, edges[2].z);*/

  auto det = c0 * v0.w + c1 * v1.w + c2 * v2.w;
  if (det < 0) {
    edges[0].x *= -1.0f;
    edges[0].y *= -1.0f;
    edges[0].z *= -1.0f;
    edges[1].x *= -1.0f;
    edges[1].y *= -1.0f;
    edges[1].z *= -1.0f;
    edges[2].x *= -1.0f;
    edges[2].y *= -1.0f;
    edges[2].z *= -1.0f;
  }

  return (det != 0);
}

static void EdgeToFixed(rast_edge_t out[3], vec3d_f_t in[3]) {
  // Normalize the matrix
  auto max_ab = std::max({std::abs(in[0].x), std::abs(in[1].x), std::abs(in[2].x),
                          std::abs(in[0].y), std::abs(in[1].y), std::abs(in[2].y)});

  #define NORMALIZE(x, y, z) { auto t = 1.0 / max_ab; x *= t; y *= t; z *= t; }
  NORMALIZE(in[0].x, in[0].y, in[0].z)
  NORMALIZE(in[1].x, in[1].y, in[1].z)
  NORMALIZE(in[2].x, in[2].y, in[2].z)  

  // Convert the edge equation to fixedpoint
  out[0] = {fixed16_t(in[0].x), fixed16_t(in[0].y), fixed16_t(in[0].z)};
  out[1] = {fixed16_t(in[1].x), fixed16_t(in[1].y), fixed16_t(in[1].z)};
  out[2] = {fixed16_t(in[2].x), fixed16_t(in[2].y), fixed16_t(in[2].z)};

  //printf("*** out0=(%d, %d, %d)\n", outs[0].x.data(), outs[0].y.data(), outs[0].z.data());
  //printf("*** out1=(%d, %d, %d)\n", outs[1].x.data(), outs[1].y.data(), outs[1].z.data());
  //printf("*** out2=(%d, %d, %d)\n", outs[2].x.data(), outs[2].y.data(), outs[2].z.data());
}

// traverse model primitives and do tile assignment
uint32_t Binning(std::vector<uint8_t>& tilebuf, 
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, CGLTrace::vertex_t>& vertices,
                 const std::vector<CGLTrace::primitive_t>& primitives,                 
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileSize) {

  uint32_t tileLogSize = log2ceil(tileSize);

  std::map<uint32_t, std::vector<uint32_t>> tiles;

  std::vector<rast_prim_t> rast_prims;
  rast_prims.reserve(primitives.size());

  rast_bbox_t global_bbox{-1u, 0, -1u, 0};

  uint32_t num_prims = 0;

  #define POS_TO_V2D(d, s) \
      d.x = s.x; \
      d.y = s.y

  #define POS_TO_V4D(d, s) \
    d.x = s.x; \
    d.y = s.y; \
    d.z = s.z; \
    d.w = s.w
  
  for (auto& primitive : primitives) {
    // get primitive vertices
    auto& v0 = vertices.at(primitive.i0);
    auto& v1 = vertices.at(primitive.i1);
    auto& v2 = vertices.at(primitive.i2);

    vec4d_f_t p0, p1, p2;
    POS_TO_V4D (p0, v0.pos);
    POS_TO_V4D (p1, v1.pos);
    POS_TO_V4D (p2, v2.pos);

    vec3d_f_t edges[3];
    rast_bbox_t bbox;
    vec4d_f_t ps0, ps1, ps2;

    {
      vec4d_f_t ph0, ph1, ph2;
      
      // Convert position from clip to 2D homogenous device space      
      ClipToHDC(&ph0, p0, 0, width, 0, height, near, far);
      ClipToHDC(&ph1, p1, 0, width, 0, height, near, far);
      ClipToHDC(&ph2, p2, 0, width, 0, height, near, far);

      // Calculate edge equation
      if (!EdgeEquation(edges, ph0, ph1, ph2)) {
        // reject degenerate triangles
        printf("warning: degenerate primitive...\n");
        continue;
      }
    }       

    {
      // Convert position from clip to screen space      
      ClipToScreen(&ps0, p0, 0, width, 0, height, near, far);
      ClipToScreen(&ps1, p1, 0, width, 0, height, near, far);
      ClipToScreen(&ps2, p2, 0, width, 0, height, near, far);

      // Calculate bounding box
      vec2d_f_t q0, q1, q2;
      POS_TO_V2D (q0, ps0);
      POS_TO_V2D (q1, ps1);
      POS_TO_V2D (q2, ps2);

      //printf("*** screen position: v0=(%f, %f), v1=(%f, %f), v2=(%f, %f)\n", q0.x, q0.y, q1.x, q1.y, q2.x, q2.y);

      rect_f_t tmp;
      CalcBoundingBox(&tmp, q0, q1, q2);
      tmp.left   = floor(tmp.left + 0.5f);
      tmp.right  = ceil(tmp.right + 0.5f);
      tmp.top    = floor(tmp.top + 0.5f);
      tmp.bottom = ceil(tmp.bottom + 0.5f);
      bbox.left   = std::max<int32_t>(0,      tmp.left);
      bbox.right  = std::min<int32_t>(width,  tmp.right);
      bbox.top    = std::max<int32_t>(0,      tmp.top);
      bbox.bottom = std::min<int32_t>(height, tmp.bottom);

      //printf("*** bbpx=(%f, %f, %f, %f)\n", tmp.left, tmp.right, tmp.top, tmp.bottom);

      global_bbox.left   = std::min<uint32_t>(bbox.left, global_bbox.left);
      global_bbox.right  = std::max<uint32_t>(bbox.right, global_bbox.right);
      global_bbox.top    = std::min<uint32_t>(bbox.top, global_bbox.top);
      global_bbox.bottom = std::max<uint32_t>(bbox.bottom, global_bbox.bottom);
    }

    uint32_t p;

    {
      #define ATTRIBUTE_DELTA(d, x0, x1, x2) \
        d.x = fixed24_t(x0 - x2); \
        d.y = fixed24_t(x1 - x2); \
        d.z = fixed24_t(x2)

      rast_prim_t rast_prim;
      EdgeToFixed(rast_prim.edges, edges);
         
      ATTRIBUTE_DELTA (rast_prim.attribs.z, ps0.z, ps1.z, ps2.z);
      ATTRIBUTE_DELTA (rast_prim.attribs.r, v0.color.r, v1.color.r, v2.color.r);
      ATTRIBUTE_DELTA (rast_prim.attribs.g, v0.color.g, v1.color.g, v2.color.g);
      ATTRIBUTE_DELTA (rast_prim.attribs.b, v0.color.b, v1.color.b, v2.color.b);
      ATTRIBUTE_DELTA (rast_prim.attribs.a, v0.color.a, v1.color.a, v2.color.a);
      ATTRIBUTE_DELTA (rast_prim.attribs.u, v0.texcoord.u, v1.texcoord.u, v2.texcoord.u);
      ATTRIBUTE_DELTA (rast_prim.attribs.v, v0.texcoord.v, v1.texcoord.v, v2.texcoord.v);

      p = rast_prims.size();
      rast_prims.push_back(rast_prim);      
    }

    // Calculate min/max tile positions
    auto tileSize = 1 << tileLogSize;
    auto minTileX = bbox.left >> tileLogSize;
    auto minTileY = bbox.top >> tileLogSize;
    auto maxTileX = (bbox.right + tileSize - 1) >> tileLogSize;
    auto maxTileY = (bbox.bottom + tileSize - 1) >> tileLogSize;

    // Starting tile coordinates
    auto X = minTileX << tileLogSize;
    auto Y = minTileY << tileLogSize;

    // Add tile corner edge offsets
    float extents[3];
    extents[0] = calcEdgeExtents(edges[0]);
    extents[1] = calcEdgeExtents(edges[1]);
    extents[2] = calcEdgeExtents(edges[2]);

    // Evaluate edge equation for the starting tile
    auto e0 = evalEdgeFunction(edges[0], X, Y);
    auto e1 = evalEdgeFunction(edges[1], X, Y);
    auto e2 = evalEdgeFunction(edges[2], X, Y);

    // traverse covered tiles
    for (uint32_t ty = minTileY; ty < maxTileY; ++ty) {
      auto ee0 = e0;
      auto ee1 = e1;
      auto ee2 = e2;
      for (uint32_t tx = minTileX; tx < maxTileX; ++tx) {
        // check if tile overlap triangle    
        auto test0 = ee0 + (extents[0] * tileSize);
        auto test1 = ee1 + (extents[1] * tileSize);
        auto test2 = ee2 + (extents[2] * tileSize);
        auto trivial_reject = (test0 < 0.0f) || (test1 < 0.0f) || (test2 < 0.0f);
        if (!trivial_reject) {
          // assign primitive to tile
          //auto x = tx << tileLogSize;
          //auto y = ty << tileLogSize;
          //printf("*** Tile (%d,%d) :\n", x, y);
          uint32_t tile_id = (ty << 16) | tx;
          tiles[tile_id].push_back(p);
          ++num_prims;
        }
        // update edge equation x components
        ee0 += edges[0].x * tileSize;
        ee1 += edges[1].x * tileSize;
        ee2 += edges[2].x * tileSize;
      }
      // update edge equation y components
      e0 += edges[0].y * tileSize;
      e1 += edges[1].y * tileSize;
      e2 += edges[2].y * tileSize;
    }
  }

  {
    primbuf.resize(rast_prims.size() * sizeof(rast_prim_t));
    memcpy(primbuf.data(), rast_prims.data(), primbuf.size());
  }
  
  {
    tilebuf.resize(tiles.size() * sizeof(rast_tile_header_t) + num_prims * sizeof(uint32_t));
    auto tile_data = tilebuf.data();
    for (auto it : tiles) {
      rast_tile_header_t header{it.first, (uint32_t)it.second.size()};
      *(rast_tile_header_t*)(tile_data) = header;
      tile_data += sizeof(rast_tile_header_t);
      memcpy(tile_data, it.second.data(), it.second.size() * sizeof(uint32_t));
      tile_data += it.second.size() * sizeof(uint32_t);
    }
  }

  //printf("Binning bounding box={l=%d, r=%d, t=%d, b=%d}\n", global_bbox.left, global_bbox.right, global_bbox.top, global_bbox.bottom);

  return tiles.size();
}

std::string getFileExt(const std::string& str) {
  auto i = str.rfind('.');
  if (i != std::string::npos) {
    return str.substr(i+1);
  }
  return("");
}

bool iequals(const std::string& a, const std::string& b) {
  auto sz = a.size();
  if (b.size() != sz)
    return false;
  for (size_t i = 0; i < sz; ++i) {
    if (tolower(a[i]) != tolower(b[i]))
      return false;
  }
  return true;
}

int LoadImage(const char *filename,
              ePixelFormat format, 
              std::vector<uint8_t> &pixels, 
              uint32_t *width,
              uint32_t *height) {
  uint32_t img_width;
  uint32_t img_height;
  uint32_t img_bpp;

  auto ext = getFileExt(filename);
  if (iequals(ext, "tga")) {
    int ret = LoadTGA(filename, pixels, &img_width, &img_height, &img_bpp);
    if (ret)
      return ret;
  } else 
  if (iequals(ext, "png")) {
    int ret = LoadPNG(filename, pixels, &img_width, &img_height, &img_bpp);
    if (ret)
      return ret;
  } else 
  if (iequals(ext, "bmp")) {
    int ret = LoadBMP(filename, pixels, &img_width, &img_height, &img_bpp);
    if (ret)
      return ret;
  } else {
    std::cerr << "invalid file extension: " << ext << "!" << std::endl;
    return -1;
  }

  ePixelFormat img_format;  
  switch (img_bpp) {
  case 1: 
    img_format = FORMAT_A8;
    break;
  case 2: 
    img_format = FORMAT_R5G6B5;
    break;
  case 3: 
    img_format = FORMAT_R8G8B8; 
    break;
  case 4: 
    img_format = FORMAT_A8R8G8B8; 
    break;
  default:
    std::abort();            
  }

  if (img_format != format) {
    // format conversion to RGBA
    std::vector<uint8_t> staging;    
    int ret = ConvertImage(staging, format, pixels.data(), img_format, img_width, img_height, img_width * img_bpp);
    if (ret)
      return ret;
    pixels.swap(staging);
  }

  *width  = img_width;
  *height = img_height;
  
  return 0;
}

int SaveImage(const char *filename,
              ePixelFormat format,
              const uint8_t* pixels, 
              uint32_t width,
              uint32_t height,
              int32_t pitch) {
  uint32_t bpp = Format::GetInfo(format).BytePerPixel;
  auto ext = getFileExt(filename);
  if (iequals(ext, "tga")) {
    return SaveTGA(filename, pixels, width, height, bpp, pitch);
  } else 
  if (iequals(ext, "png")) {
    return SavePNG(filename, pixels, width, height, bpp, pitch);
  } else 
  if (iequals(ext, "bmp")) {
    return SaveBMP(filename, pixels, width, height, bpp, pitch);
  } else {
    std::cerr << "invalid file extension: " << ext << "!" << std::endl;
    return -1;
  }

  return 0;
}

void dump_image(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height, uint32_t bpp) {
  assert(width * height * bpp == pixels.size());
  const uint8_t* pixel_bytes = pixels.data();
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t pixel32 = 0;
      for (uint32_t b = 0; b < bpp; ++b) {
        uint32_t pixel8 = *pixel_bytes++;
        pixel32 |= pixel8 << (b * 8);
      }
      if (x) std::cout << ", ";
      std::cout << std::hex << std::setw(bpp * 2) << std::setfill('0') << pixel32;
    }
    std::cout << std::endl;
  }
}

int CompareImages(const char* filename1, 
                  const char* filename2, 
                  cocogfx::ePixelFormat format) {
  int ret;
  std::vector<uint8_t> image1_bits;  
  uint32_t image1_width; 
  uint32_t image1_height;

  std::vector<uint8_t> image2_bits;  
  uint32_t image2_width; 
  uint32_t image2_height;
    
  ret = LoadImage(filename1, format, image1_bits, &image1_width, &image1_height);  
  if (ret)
    return ret;

  ret = LoadImage(filename2, format, image2_bits, &image2_width, &image2_height);  
  if (ret)
    return ret;

  if (image1_bits.size() != image2_bits.size())
    return -1;

  if (image1_width != image2_width)
    return -1;

  if (image1_height != image2_height)
    return -1;

  int errors = 0;
  {
    auto convert_from = Format::GetConvertFrom(format, true);
    auto bpp = Format::GetInfo(format).BytePerPixel;
    auto pixels1 = image1_bits.data();
    auto pixels2 = image2_bits.data();
    for (uint32_t y = 0; y < image1_height; ++y) {
      for (uint32_t x = 0; x < image1_width; ++x) {
        auto color1 = convert_from(pixels1);
        auto color2 = convert_from(pixels2);
        if (color1 != color2) {
          printf("Error: pixel mismatch at (%d, %d), actual=0x%x, expected=0x%x\n", x, y, color1.value, color2.value);
          ++errors;
        }
        pixels1 += bpp;
        pixels2 += bpp;    
      }
    }
  }

  return errors;
}

uint32_t toVXFormat(ePixelFormat format) {
  switch (format) {
  case FORMAT_A8R8G8B8: return TEX_FORMAT_A8R8G8B8; break;
  case FORMAT_R5G6B5: return TEX_FORMAT_R5G6B5; break;
  case FORMAT_A1R5G5B5: return TEX_FORMAT_A1R5G5B5; break;
  case FORMAT_A4R4G4B4: return TEX_FORMAT_A4R4G4B4; break;
  case FORMAT_A8L8: return TEX_FORMAT_A8L8; break;
  case FORMAT_L8: return TEX_FORMAT_L8; break;
  case FORMAT_A8: return TEX_FORMAT_A8; break;
  default:
    std::cout << "Error: invalid format: " << format << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXCompare(CGLTrace::ecompare compare) {
  switch (compare) {
  case CGLTrace::COMPARE_NEVER: return ROP_DEPTH_FUNC_NEVER; break;
  case CGLTrace::COMPARE_LESS: return ROP_DEPTH_FUNC_LESS; break;
  case CGLTrace::COMPARE_EQUAL: return ROP_DEPTH_FUNC_EQUAL; break;
  case CGLTrace::COMPARE_LEQUAL: return ROP_DEPTH_FUNC_LEQUAL; break;
  case CGLTrace::COMPARE_GREATER: return ROP_DEPTH_FUNC_GREATER; break;
  case CGLTrace::COMPARE_NOTEQUAL: return ROP_DEPTH_FUNC_NOTEQUAL; break;
  case CGLTrace::COMPARE_GEQUAL: return ROP_DEPTH_FUNC_GEQUAL; break;
  case CGLTrace::COMPARE_ALWAYS: return ROP_DEPTH_FUNC_ALWAYS; break;
  default:
    std::cout << "Error: invalid compare function: " << compare << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXStencilOp(CGLTrace::eStencilOp op) {
  switch (op) {
  case CGLTrace::STENCIL_KEEP: return ROP_STENCIL_OP_KEEP; break;
  case CGLTrace::STENCIL_REPLACE: return ROP_STENCIL_OP_REPLACE; break;
  case CGLTrace::STENCIL_INCR: return ROP_STENCIL_OP_INCR; break;
  case CGLTrace::STENCIL_DECR: return ROP_STENCIL_OP_DECR; break;
  case CGLTrace::STENCIL_ZERO: return ROP_STENCIL_OP_ZERO; break;
  case CGLTrace::STENCIL_INVERT: return ROP_STENCIL_OP_INVERT; break;
  default:
    std::cout << "Error: invalid stencil operation: " << op << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXBlendFunc(CGLTrace::eBlendOp op) {
  switch (op) {
  case CGLTrace::BLEND_ZERO: return ROP_BLEND_FUNC_ZERO;
  case CGLTrace::BLEND_ONE: return ROP_BLEND_FUNC_ONE;
  case CGLTrace::BLEND_SRC_COLOR: return ROP_BLEND_FUNC_SRC_RGB;
  case CGLTrace::BLEND_ONE_MINUS_SRC_COLOR: return ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB;
  case CGLTrace::BLEND_SRC_ALPHA: return ROP_BLEND_FUNC_SRC_A;
  case CGLTrace::BLEND_ONE_MINUS_SRC_ALPHA: return ROP_BLEND_FUNC_ONE_MINUS_SRC_A;
  case CGLTrace::BLEND_DST_ALPHA: return ROP_BLEND_FUNC_DST_A;
  case CGLTrace::BLEND_ONE_MINUS_DST_ALPHA: return ROP_BLEND_FUNC_ONE_MINUS_DST_A;
  case CGLTrace::BLEND_DST_COLOR: return ROP_BLEND_FUNC_DST_RGB;
  case CGLTrace::BLEND_ONE_MINUS_DST_COLOR: return ROP_BLEND_FUNC_ONE_MINUS_DST_RGB;
  case CGLTrace::BLEND_SRC_ALPHA_SATURATE: return ROP_BLEND_FUNC_ALPHA_SAT;
  default:
    std::cout << "Error: invalid blend function: " << op << std::endl;
    exit(1);
  }
  return 0;
}