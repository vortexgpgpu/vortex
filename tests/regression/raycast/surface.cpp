#include "surface.h"
#include <cstring>
#include <math.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_HDR
#include "stb_image.h"

Surface::Surface(int w, int h, uint8_t *b) : pixels_(b), width_(w), height_(h) {}

Surface::Surface(int w, int h) : width_(w), height_(h) {
  pixels_ = (uint8_t *)std::aligned_alloc(64, w * h * sizeof(uint32_t));
  ownBuffer_ = true; // needs to be deleted in destructor
}

Surface::Surface(const char *file) : pixels_(0), width_(0), height_(0) {
  FILE *f = fopen(file, "rb");
  if (!f) {
    fprintf(stderr, "File not found: %s", file);
    std::abort();
  }
  fclose(f);
  this->loadImage(file);
}

void Surface::loadImage(const char *file) {
  if (stbi_is_hdr(file)) {
    float *data = stbi_loadf(file, (int*)&width_, (int*)&height_, nullptr, 3);
    if (data) {
      const int s = width_ * height_;
      auto dest = (float *)std::aligned_alloc(64, s * 3 * sizeof(float));
      for (int i = 0; i < 3 * s; i++) {
        dest[i] = sqrtf(data[i]);
      }
      pixels_ = (uint8_t*)dest;
    }
    stbi_image_free(data);
    bpp_ = 3 * sizeof(float);
  } else {
    uint8_t *data = stbi_load(file, (int*)&width_, (int*)&height_, nullptr, 3);
    if (data) {
      const int s = width_ * height_;
      auto dest = (uint32_t *)std::aligned_alloc(64, s * sizeof(uint32_t));
      for (int i = 0; i < s; i++) {
        dest[i] = (data[i * 3 + 0] << 16) + (data[i * 3 + 1] << 8) + data[i * 3 + 2];
      }
      pixels_ = (uint8_t*)dest;
    }
    stbi_image_free(data);
    bpp_ = sizeof(uint32_t);
  }
  ownBuffer_ = true; // needs to be deleted in destructor
}

Surface::~Surface() {
  if (ownBuffer_) {
    free(pixels_); // free only if we allocated the buffer ourselves
  }
}
