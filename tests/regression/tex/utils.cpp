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



int GenerateMipmaps(std::vector<uint8_t>& dst_pixels,
                    std::vector<uint32_t>& mip_offsets,
                    const std::vector<uint8_t>& src_pixels,
                    ePixelFormat format,
                    uint32_t src_width,
                    uint32_t src_height) {
  std::vector<uint8_t> src_staging, dst_staging;
  const std::vector<uint8_t> *pSrcPixels;
  std::vector<uint8_t> *pDstPixels;

  // convert source image if needed
  bool need_conversion = (format != FORMAT_A8R8G8B8);
  if (need_conversion) {
    ConvertImage(src_staging, src_pixels, src_width, src_height, format, FORMAT_A8R8G8B8);
    pSrcPixels = &src_staging;
    pDstPixels = &dst_staging;
  } else {
    pSrcPixels = &src_pixels;
    pDstPixels = &dst_pixels;
  }

  uint32_t src_logwidth  = log2ceil(src_width);
  uint32_t src_logheight = log2ceil(src_height);
  uint32_t max_lod       = std::max(src_logwidth, src_logheight) + 1;

  mip_offsets.resize(max_lod);

  // Calculate mipmaps buffer size
  uint32_t dst_height = 1;
  uint32_t dst_width = 0;
  for (uint32_t lod = 0, w = src_width, h = src_height; lod < max_lod; ++lod) {
    assert((w > 0) || (w > 0));
    uint32_t pw = std::max<int>(w, 1);
    uint32_t ph = std::max<int>(h, 1);
    mip_offsets.at(lod) = dst_width;
    dst_width += pw * ph;
    w >>= 1;
    h >>= 1;
  }

  // allocate mipmap
  pDstPixels->resize(dst_width * 4);

  // generate mipmaps  
  {
    auto pSrc = reinterpret_cast<const uint32_t*>(pSrcPixels->data());
    auto pDst = reinterpret_cast<uint32_t*>(pDstPixels->data());

    // copy level 0
    memcpy(pDst, pSrc, pSrcPixels->size());
    assert(pSrcPixels->size() == 4 * src_width * src_height);
    pSrc = pDst;
    pDst += src_width * src_height;    

    // copy lower levels
    for (uint32_t lod = 1, w = (src_width/2), h = (src_height/2); lod < max_lod;) {
      assert((w > 0) || (w > 0));
      uint32_t pw = std::max<int>(w, 1);
      uint32_t ph = std::max<int>(h, 1);
      for (uint32_t y = 0; y < pw; ++y) {
        auto v0 = 2 * y;
        auto v1 = 2 * y + ((ph > 1) ? 1 : 0);
        auto pSrc0 = pSrc + v0 * (2 * pw);
        auto pSrc1 = pSrc + v1 * (2 * pw);

        for (uint32_t x = 0; x <pw; ++x) {
          auto u0 = 2 * x;
          auto u1 = 2 * x + ((pw > 1) ? 1 : 0);

          auto c00 = Format::ConvertFrom<FORMAT_A8R8G8B8, false>(pSrc0 + u0);
          auto c01 = Format::ConvertFrom<FORMAT_A8R8G8B8, false>(pSrc0 + u1);
          auto c10 = Format::ConvertFrom<FORMAT_A8R8G8B8, false>(pSrc1 + u0);
          auto c11 = Format::ConvertFrom<FORMAT_A8R8G8B8, false>(pSrc1 + u1);

          const ColorARGB color((c00.a + c01.a + c10.a + c11.a+2) >> 2,
                                (c00.r + c01.r + c10.r + c11.r+2) >> 2,
                                (c00.g + c01.g + c10.g + c11.g+2) >> 2,
                                (c00.b + c01.b + c10.b + c11.b+2) >> 2);
                                
          uint32_t ncolor;
          Format::ConvertTo<FORMAT_A8R8G8B8>(&ncolor, color);
          pDst[x + y * pw] = ncolor;
        }
      } 
      ++lod; 
      pSrc = pDst;
      pDst += pw * ph;
      w >>= 1;
      h >>= 1;  
    }
    assert((pDst - reinterpret_cast<uint32_t*>(pDstPixels->data())) == dst_width);
  }

  // convert destination image if needed
  if (need_conversion) {
    ConvertImage(dst_staging, dst_staging, dst_width, dst_height, FORMAT_A8R8G8B8, format);
  }

  uint32_t bpp = Format::GetInfo(format).BytePerPixel;
  for (auto& offset : mip_offsets) {
    offset *= bpp;
  }

  return 0;
}