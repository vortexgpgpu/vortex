#include "utils.h"
#include <assert.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <unordered_map>
#include <cocogfx/include/tga.hpp>
#include <cocogfx/include/png.hpp>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

using namespace cocogfx;

using fixed16_t  = TFixed<16>;

using vec2d_f_t  = TVector2<float>;
using vec2d_fx_t = TVector2<fixed16_t>;

using vec4d_f_t  = TVector4<float>;

using rect_f_t   = TRect<float>;

static fixed16_t fxZero(0);
static fixed16_t fxHalf(0.5f);

// Evaluate edge function
static fixed16_t evalEdgeFunction(const rast_edge_t& e, uint32_t x, uint32_t y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// Calculate the edge extents for tile corners
static fixed16_t calcEdgeExtents(const rast_edge_t& e, uint32_t logTileSize) {
  vec2d_fx_t corners[4] = {{fxZero, fxZero},  // 00
                            {e.x,    fxZero},  // 10
                            {fxZero, e.y},     // 01
                            {e.x,    e.y}};    // 11
  auto i = (e.y >= fxZero) ? ((e.x >= fxZero) ? 3 : 2) : (e.x >= fxZero) ? 1 : 0;
  return (corners[i].x + corners[i].y) << logTileSize;
}

static float EdgeEquation(rast_edge_t edges[3], 
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

  // Normalize the matrix
  #define NORMALIZE(x, y, z) { auto t = 1.0 / (std::abs(x) + std::abs(y)); x *= t; y *= t; z *= t; }
  NORMALIZE(a0, b0, c0)
  NORMALIZE(a1, b1, c1)
  NORMALIZE(a2, b2, c2)

  // Convert the edge equation to fixedpoint
  edges[0] = {fixed16_t(a0), fixed16_t(b0), fixed16_t(c0)};
  edges[1] = {fixed16_t(a1), fixed16_t(b1), fixed16_t(c1)};
  edges[2] = {fixed16_t(a2), fixed16_t(b2), fixed16_t(c2)};

  /*printf("E0.x=%f, E0.y=%f, E0.z=%f, E1.x=%f, E1.y=%f, E1.z=%f, E2.x=%f, E2.y=%f, E2.z=%f\n", 
      float(edges[0].x), float(edges[0].y), float(edges[0].z),
      float(edges[1].x), float(edges[1].y), float(edges[1].z),
      float(edges[2].x), float(edges[2].y), float(edges[2].z));*/

  auto det = c0 * v0.w + c1 * v1.w + c2 * v2.w;

  return det;
}

static void ColorToFloat(float out[4], uint32_t color) {
  out[0] = ((color >>  0) & 0xFF) / 255.0f;
  out[1] = ((color >>  8) & 0xFF) / 255.0f;
  out[2] = ((color >> 16) & 0xFF) / 255.0f;
  out[3] = ((color >> 24) & 0xFF) / 255.0f;
}

// traverse model primitives and do tile assignment
uint32_t Binning(std::vector<uint8_t>& tilebuf, 
                 std::vector<uint8_t>& primbuf,
                 const model_t& model, 
                 uint32_t width,
                 uint32_t height,
                 uint32_t tileSize) {

  uint32_t logTileSize = log2ceil(tileSize);

  std::unordered_map<uint32_t, std::vector<uint32_t>> tiles;

  std::vector<rast_prim_t> rast_prims;
  rast_prims.reserve(model.primitives.size());

  uint32_t num_prims = 0;
  
  for (auto& primitive : model.primitives) {
    // get primitive vertices
    auto& v0 = model.vertives.at(primitive.i0);
    auto& v1 = model.vertives.at(primitive.i1);
    auto& v2 = model.vertives.at(primitive.i2);

    auto& p0 = *(vec4d_f_t*)&v0;
    auto& p1 = *(vec4d_f_t*)&v1;
    auto& p2 = *(vec4d_f_t*)&v2;

    rast_edge_t edges[3];
    rast_bbox_t bbox;

    {
      // Convert position from clip to 2D homogenous device space
      vec4d_f_t q0, q1, q2;
      ClipTo2DH(&q0, p0, width, height);
      ClipTo2DH(&q1, p1, width, height);
      ClipTo2DH(&q2, p2, width, height);

      // Calculate edge equation
      auto det = EdgeEquation(edges, q0, q1, q2);
      if (det <= 0) {
        // reject back-facing or degenerate triangles
        continue;
      }
    }   

    {
      // Convert position from clip to screen space
      vec4d_f_t q0, q1, q2;
      ClipToScreen(&q0, p0, width, height);
      ClipToScreen(&q1, p1, width, height);
      ClipToScreen(&q2, p2, width, height);

      // Calculate bounding box 
      rect_f_t tmp;
      auto _q0 = (vec2d_f_t*)&q0;
      auto _q1 = (vec2d_f_t*)&q1;
      auto _q2 = (vec2d_f_t*)&q2;
      CalcBoundingBox(&tmp, *_q0, *_q1, *_q2);
      bbox.left   = std::max<int32_t>(0, tmp.left);
      bbox.right  = std::min<int32_t>(width, tmp.right);
      bbox.top    = std::max<int32_t>(0, tmp.top);
      bbox.bottom = std::min<int32_t>(height, tmp.bottom);
    }

    uint32_t p;

    {
      #define INTERPOLATE_DELTA(dx, x0, x1, x2) \
        dx.x = fixed23_t(x0 - x2); \
        dx.y = fixed23_t(x1 - x2); \
        dx.z = fixed23_t(x2)

      rast_prim_t rast_prim;
      rast_prim.edges[0] = edges[0];
      rast_prim.edges[1] = edges[1];
      rast_prim.edges[2] = edges[2];
      
      rast_prim.bbox = bbox;

      float colors[3][4];
      ColorToFloat(colors[0], v0.c);
      ColorToFloat(colors[1], v1.c);
      ColorToFloat(colors[2], v2.c);
      
      INTERPOLATE_DELTA(rast_prim.attribs.z, v0.z, v1.z, v2.z);
      INTERPOLATE_DELTA(rast_prim.attribs.r, colors[0][0], colors[1][0], colors[2][0]);
      INTERPOLATE_DELTA(rast_prim.attribs.g, colors[0][1], colors[1][1], colors[2][1]);
      INTERPOLATE_DELTA(rast_prim.attribs.b, colors[0][2], colors[1][2], colors[2][2]);
      INTERPOLATE_DELTA(rast_prim.attribs.a, colors[0][3], colors[1][3], colors[2][3]);      
      INTERPOLATE_DELTA(rast_prim.attribs.u, v0.u, v1.u, v2.u);
      INTERPOLATE_DELTA(rast_prim.attribs.v, v0.v, v1.v, v2.v);

      p = rast_prims.size();
      rast_prims.push_back(rast_prim);      
    }

    // Calculate min/max tile positions
    auto tileSize = 1 << logTileSize;
    auto minTileX = bbox.left >> logTileSize;
    auto minTileY = bbox.top >> logTileSize;
    auto maxTileX = (bbox.right + tileSize - 1) >> logTileSize;
    auto maxTileY = (bbox.bottom + tileSize - 1) >> logTileSize;

    // Starting tile coordinates
    auto X = minTileX << logTileSize;
    auto Y = minTileY << logTileSize;

    // Add tile corner edge offsets
    fixed16_t extents[3];
    extents[0] = calcEdgeExtents(edges[0], logTileSize);
    extents[1] = calcEdgeExtents(edges[1], logTileSize);
    extents[2] = calcEdgeExtents(edges[2], logTileSize);

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
        if (((ee0 + extents[0]).data() 
           | (ee1 + extents[1]).data()
           | (ee2 + extents[2]).data()) >= 0) {
          // assign primitive to tile
          uint32_t tile_id = (ty << 16) | tx;
          tiles[tile_id].push_back(p);
          ++num_prims;
        }

        // update edge equation x components
        ee0 += edges[0].x << logTileSize;
        ee1 += edges[1].x << logTileSize;
        ee2 += edges[2].x << logTileSize;
      }
      // update edge equation y components
      e0 += edges[0].y << logTileSize;
      e1 += edges[1].y << logTileSize;
      e2 += edges[2].y << logTileSize;
    }
  }

  {
    primbuf.reserve(rast_prims.size()  * sizeof(rast_prim_t));
    memcpy(primbuf.data(), rast_prims.data(), primbuf.size());
  }
  
  {
    tilebuf.reserve(tiles.size() * sizeof(rast_tile_header_t) + num_prims * sizeof(uint32_t));
    auto tile_data = tilebuf.data();
    for (auto it : tiles) {
      rast_tile_header_t header{it.first, (uint32_t)it.second.size()};
      *(rast_tile_header_t*)(tile_data) = header;
      tile_data += sizeof(rast_tile_header_t);
      memcpy(tile_data, it.second.data(), it.second.size() * sizeof(uint32_t));
      tile_data += it.second.size() * sizeof(uint32_t);
    }
  }

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
    img_format = FORMAT_A1R5G5B5;
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
    int ret = ConvertImage(staging, format, pixels, img_format, img_width, img_height, img_width * img_bpp);
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
              const std::vector<uint8_t> &pixels, 
              uint32_t width,
              uint32_t height) {
  uint32_t bpp = Format::GetInfo(format).BytePerPixel;
  auto ext = getFileExt(filename);
  if (iequals(ext, "tga")) {
    return SaveTGA(filename, pixels, width, height, bpp);
  } else 
  if (iequals(ext, "png")) {
    return SavePNG(filename, pixels, width, height, bpp);
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