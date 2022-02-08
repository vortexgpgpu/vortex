#include "utils.h"
#include <assert.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cocogfx/include/tga.h>
#include <cocogfx/include/png.h>

using namespace cocogfx;

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
  uint32_t bpp = GetInfo(format).BytePerPixel;
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