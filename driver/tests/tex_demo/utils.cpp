#include "utils.h"
#include <fstream>
#include <assert.h>
#include "format.h"

struct __attribute__((__packed__)) tga_header_t {
  int8_t idlength;
  int8_t colormaptype;
  int8_t imagetype;
  int16_t colormaporigin;
  int16_t colormaplength;
  int8_t colormapdepth;
  int16_t xoffset;
  int16_t yoffset;
  int16_t width;
  int16_t height;
  int8_t bitsperpixel;
  int8_t imagedescriptor;
};

int LoadTGA(const char *filename, 
            std::vector<uint8_t> &pixels, 
            uint32_t *width, 
            uint32_t *height) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "couldn't open file: " << filename << "!" << std::endl;
    return -1;
  }

  tga_header_t header;
  ifs.read(reinterpret_cast<char *>(&header), sizeof(tga_header_t));
  if (ifs.fail()) {
    std::cerr << "invalid TGA file header!" << std::endl;
    return -1;
  }

  if (header.imagetype != 2) {
    std::cerr << "unsupported TGA encoding format!" << std::endl;
    return -1;
  }

  ifs.seekg(header.idlength, std::ios::cur); // skip string
  if (ifs.fail()) {
    std::cerr << "invalid TGA file!" << std::endl;
    return -1;
  }

  switch (header.bitsperpixel) {
  case 16:
  case 24:
  case 32: {
    auto stride = header.bitsperpixel / 8;
    std::vector<uint8_t> staging(stride * header.width * header.height);

    // Read pixels data
    ifs.read((char*)staging.data(), staging.size());
    if (ifs.fail()) {
      std::cerr << "invalid TGA file!" << std::endl;
      return -1;
    }

    // format conversion to RGBA
    pixels.resize(4 * header.width * header.height);
    const uint8_t* src_bytes = staging.data();
    uint32_t* dst_bytes = (uint32_t*)pixels.data();
    for (const uint8_t* const src_end = src_bytes + staging.size(); 
         src_bytes != src_end; 
         src_bytes += stride) {
      ColorARGB color;        
      switch (stride) {
      case 2: 
        color = Format::ConvertFrom<FORMAT_A1R5G5B5, true>(src_bytes);         
        break;
      case 3: 
        color = Format::ConvertFrom<FORMAT_R8G8B8, true>(src_bytes); 
        break;
      case 4: 
        color = Format::ConvertFrom<FORMAT_A8R8G8B8, true>(src_bytes); 
        break;
      default:
        std::abort();            
      }
      *dst_bytes++ = color;
    }
    break;
  }
  default:
    std::cerr << "unsupported TGA bitsperpixel!" << std::endl;
    return -1;
  } 

  *width = header.width;
  *height = header.height;

  return 0;
}

int SaveTGA(const char *filename, 
            const std::vector<uint8_t> &pixels, 
            uint32_t width, 
            uint32_t height, 
            uint32_t bpp) {              
  std::ofstream ofs(filename, std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "couldn't create file: " << filename << "!" << std::endl;
    return -1;
  }

  if (bpp < 2 || bpp > 4) {        
    std::cerr << "unsupported pixel stride: " << bpp << "!" << std::endl;
    return -1;
  }

  tga_header_t header;
  header.idlength = 0;
  header.colormaptype = 0; // no palette
  header.imagetype = 2; // color mapped data
  header.colormaporigin = 0;
  header.colormaplength = 0;
  header.colormapdepth = 0;
  header.xoffset = 0;
  header.yoffset = 0;
  header.width = width;
  header.height = height;
  header.bitsperpixel = bpp * 8;
  header.imagedescriptor = 0;

  // write header
  ofs.write(reinterpret_cast<char *>(&header), sizeof(tga_header_t));

  // write pixel data
  uint32_t pitch = bpp * width;
  const uint8_t* pixel_bytes = pixels.data() + (height - 1) * pitch;
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t* pixel_row = pixel_bytes;
    for (uint32_t x = 0; x < width; ++x) {
      ofs.write((const char*)pixel_row, bpp);      
      pixel_row += bpp;
    }
    pixel_bytes -= pitch;
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