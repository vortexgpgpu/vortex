#include <cstdint>
#include <vector>
#include <iostream>

int LoadTGA(const char *filename, 
            std::vector<uint8_t> &pixels, 
            uint32_t *width,
            uint32_t *height, 
            uint32_t *bpp);

int SaveTGA(const char *filename, 
            const std::vector<uint8_t> &pixels, 
            uint32_t width,
            uint32_t height, 
            uint32_t bpp);