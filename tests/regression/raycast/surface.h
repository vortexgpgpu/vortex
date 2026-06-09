#pragma once

#include <stdint.h>

class Surface {
public:
  // constructor / destructor
  Surface() = default;
  Surface(int w, int h, uint8_t *buffer);
  Surface(int w, int h);
  Surface(const char *file);
  ~Surface();

  const uint8_t* pixels() const { return ( const uint8_t*)pixels_; }
  uint8_t* pixels() { return ( uint8_t*)pixels_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  uint32_t bpp() const { return bpp_; }
  uint32_t size() const { return width_ * height_ * bpp_; }

private:
  uint8_t *pixels_ = nullptr;
  uint32_t width_ = 0;
	uint32_t height_ = 0;
  uint32_t bpp_ = 0;
  bool     ownBuffer_ = false;

  void loadImage(const char *file);
};
