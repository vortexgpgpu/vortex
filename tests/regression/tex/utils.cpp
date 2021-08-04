#include "utils.h"
#include <assert.h>
#include <cstring>
#include "blitter.h"
#include "format.h"
#include "tga.h"
#include "lupng.h"

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
    auto image = luPngReadFile(filename, NULL);
    if (image == NULL)
      return -1;
    if (image->depth != 8 
      || (image->channels != 3 
       && image->channels != 4)) {
      luImageRelease(image, NULL);
      std::cerr << "invalid png file format!" << std::endl;  
      return -1;
    }
    pixels.resize(image->channels * image->width * image->height);
    memcpy(pixels.data(), image->data, pixels.size());
    img_width  = image->width;
    img_height = image->height;
    img_bpp    = image->channels;
    luImageRelease(image, NULL);
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
    int ret = ConvertImage(staging, pixels, img_width, img_height, img_format, format);
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
    LuImage image;
    image.width    = width;
    image.height   = height;
    image.depth    = 8;
    image.channels = bpp;
    image.data     = (uint8_t*)pixels.data();
    return luPngWriteFile(filename, &image);
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
      std::cout << std::hex << pixel32;
    }
    std::cout << std::endl;
  }
}

int CopyBuffers(SurfaceDesc &dstDesc, 
                int32_t dstOffsetX,
                int32_t dstOffsetY, 
                uint32_t copyWidth, 
                uint32_t copyHeight,
                const SurfaceDesc &srcDesc, 
                int32_t srcOffsetX,                
                int32_t srcOffsetY) {

  static const BlitTable s_blitTable;

  if ((srcOffsetX >= (int32_t)srcDesc.Width) || (srcOffsetY >= (int32_t)srcDesc.Height) ||
      (dstOffsetX >= (int32_t)dstDesc.Width) || (dstOffsetY >= (int32_t)dstDesc.Height)) {
    return -1;
  }

  if (copyWidth > dstDesc.Width) {
    copyWidth = dstDesc.Width;
  }

  if (copyWidth > srcDesc.Width) {
    copyWidth = srcDesc.Width;
  }

  if (copyHeight > dstDesc.Height) {
    copyHeight = dstDesc.Height;
  }

  if (copyHeight > srcDesc.Height) {
    copyHeight = srcDesc.Height;
  }

  return s_blitTable.get(srcDesc.Format, dstDesc.Format)(
    dstDesc, dstOffsetX, dstOffsetY, copyWidth, copyHeight, srcDesc,
    srcOffsetX, srcOffsetY);
}

int ConvertImage(std::vector<uint8_t>& dst_pixels,
                 const std::vector<uint8_t>& src_pixels,
                 uint32_t width,
                 uint32_t height,
                 ePixelFormat src_format,
                 ePixelFormat dst_format) {

  uint32_t src_pitch = Format::GetInfo(src_format).BytePerPixel * width;
  uint32_t dst_pitch = Format::GetInfo(dst_format).BytePerPixel * width;

  dst_pixels.resize(dst_pitch * height);

  SurfaceDesc srcDesc{src_format, (uint8_t*)src_pixels.data(), width, height, src_pitch};            
  SurfaceDesc dstDesc{dst_format, dst_pixels.data(), width, height, dst_pitch};

  return CopyBuffers(dstDesc, 0, 0, width, height, srcDesc, 0, 0);
}