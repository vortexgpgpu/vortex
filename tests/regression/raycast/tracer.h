#pragma once

#include "common.h"
#include "scene.h"
#include <vortex.h>

class Tracer {
public:
  Tracer(uint32_t dst_width, uint32_t dst_height,
         uint32_t samples_per_pixel,
         uint32_t max_depth, bool use_cpu = false);
  ~Tracer();

  int init(const char *kernel_file, const char* model_file, uint32_t mesh_count);

  int setup(float camera_vfov, float zoom, float3_t light_pos, float3_t light_color, float3_t ambient_color, float3_t background_color);

  int run(const char *output_file);

private:
  void render();

  uint32_t dst_width_;
  uint32_t dst_height_;
  bool use_cpu_;

  Scene *scene_ = nullptr;

  kernel_arg_t kernel_arg_ = {};        // kernel arguments
  vx_device_h device_ = nullptr;        // Vortex device handle
  vx_buffer_h args_buffer_ = nullptr;   // store kernel arguments
  vx_buffer_h krnl_buffer_ = nullptr;   // store kernel binary
  vx_buffer_h output_buffer_ = nullptr; // store output image
  vx_buffer_h triBuffer_ = nullptr;     // store Tri data
  vx_buffer_h triExBuffer_ = nullptr;   // store TriEx data
  vx_buffer_h texBuffer_ = nullptr;     // store texture data
  vx_buffer_h tlasBuffer_ = nullptr;    // store TLAS nodes
  vx_buffer_h blasBuffer_ = nullptr;    // store BLAS nodes
  vx_buffer_h bvhBuffer_ = nullptr;     // store BVH nodes
  vx_buffer_h idxBuffer_ = nullptr;     // store triangle indices
};
