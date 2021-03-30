#include "utils.h"
#include <fstream>
#include <assert.h>

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
            uint32_t *height, 
            uint32_t *bpp) {
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
  case 24:
  case 32: {
    auto stride = header.bitsperpixel / 8;
    auto pitch = header.width * stride;
    pixels.resize(header.height * pitch);

    // we are going to load the pixel data line by line
    for (int y = 0; y < header.height; ++y) {
      // Read current line of pixels
      auto line = pixels.data() + y * pitch;
      ifs.read(reinterpret_cast<char *>(line), pitch);
      if (ifs.fail()) {
        std::cerr << "invalid TGA file!" << std::endl;
        return -1;
      }

      // Because the TGA is BGR instead of RGB, we must swap RG components
      for (int i = 0; i < pitch; i += stride) {
        auto tmp = line[i];
        line[i] = line[i + 2];
        line[i + 2] = tmp;
      }
    }
    break;
  }
  default:
    std::cerr << "unsupported TGA bitsperpixel!" << std::endl;
    return -1;
  }

  *width = header.width;
  *height = header.height;
  *bpp = header.bitsperpixel / 8;

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

  ofs.write(reinterpret_cast<char *>(&header), sizeof(tga_header_t));

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