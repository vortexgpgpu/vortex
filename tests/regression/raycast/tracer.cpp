#include "tracer.h"
#define __UNIFORM__
#include "render.h"
#include <iostream>
#include <fstream>
#include <sstream>

struct material_t {
  const char* texture;
  float reflectivity;
};

const material_t materials [] = {{"ceramic.png", 0}, {"red.png", 0.5}, {"flower.png", 0.3}};

static void write_ppm(const uint8_t* output, uint32_t width, uint32_t height, const char *output_file) {
  std::ofstream ofs(output_file, std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output file: " << output_file << std::endl;
    std::abort();
  }
  ofs << "P3\n" << width << " " << height << "\n255\n";

  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      size_t idx = (height - 1 - y) * width + x;
      float r = output[idx * 4 + 0];
      float g = output[idx * 4 + 1];
      float b = output[idx * 4 + 2];
      ofs << b << " " << g << " " << r << "\n";
    }
  }
  std::cout << "Image saved to: " << output_file << std::endl;
}

static std::string resolve_path(const std::string& filename, const std::string& searchPaths) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::stringstream ss(searchPaths);
    std::string path;
    while (std::getline(ss, path, ',')) {
      if (!path.empty()) {
        std::string filePath = path + "/" + filename;
        std::ifstream ifs(filePath);
        if (ifs)
          return filePath;
      }
    }
  }
  return filename;
}

Tracer::Tracer(
  uint32_t dst_width,
  uint32_t dst_height,
  uint32_t samples_per_pixel,
  uint32_t max_depth,
  bool use_cpu)
  : dst_width_(dst_width)
  , dst_height_(dst_height)
  , use_cpu_(use_cpu)
{
  kernel_arg_.dst_width = dst_width_;
  kernel_arg_.dst_height = dst_height_;
  kernel_arg_.samples_per_pixel = samples_per_pixel;
  kernel_arg_.max_depth = max_depth;
}

Tracer::~Tracer() {
  // free allocated objects
  delete scene_;

  if (!use_cpu_) {
    // free buffers
    vx_mem_free(output_buffer_);
    vx_mem_free(args_buffer_);
    vx_mem_free(krnl_buffer_);
    vx_mem_free(triBuffer_);
    vx_mem_free(triExBuffer_);
    vx_mem_free(texBuffer_);
    vx_mem_free(tlasBuffer_);
    vx_mem_free(blasBuffer_);
    vx_mem_free(bvhBuffer_);
    vx_mem_free(idxBuffer_);
    // close device
    vx_dev_close(device_);
  }
}

int Tracer::init(const char *kernel_file, const char* model_file, uint32_t mesh_count) {
  // create meshes
  std::vector<Mesh*> meshes(mesh_count);
  for (uint32_t i = 0; i < meshes.size(); ++i) {
    auto s_model = resolve_path(std::string("assets/") + model_file, ASSETS_PATHS);
    auto s_texture = resolve_path(std::string("assets/") + materials[i].texture, ASSETS_PATHS);
    meshes[i] = new Mesh(s_model.c_str(), s_texture.c_str(), materials[i].reflectivity);
  }

  // create scene
  scene_ = new Scene(meshes);
  RT_CHECK(scene_->init());

  if (use_cpu_) {
    kernel_arg_.tri_addr = (uint64_t)scene_->tri_buf().data();
    kernel_arg_.triEx_addr = (uint64_t)scene_->triEx_buf().data();
    kernel_arg_.triIdx_addr = (uint64_t)scene_->triIdx_buf().data();
    kernel_arg_.bvh_addr = (uint64_t)scene_->bvh_nodes().data();
    kernel_arg_.tlas_addr = (uint64_t)scene_->tlas_nodes().data();
    kernel_arg_.blas_addr = (uint64_t)scene_->blas_nodes().data();
    kernel_arg_.tex_addr = (uint64_t)scene_->tex_buf().data();
  } else {
    RT_CHECK(vx_dev_open(&device_));

    // upload kernel
    RT_CHECK(vx_upload_kernel_file(device_, kernel_file, &krnl_buffer_));

    // allocate tri buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->tri_buf().size() * sizeof(tri_t), VX_MEM_READ, &triBuffer_));
    RT_CHECK(vx_mem_address(triBuffer_, &kernel_arg_.tri_addr));

    // allocate triEx buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->triEx_buf().size() * sizeof(tri_ex_t), VX_MEM_READ, &triExBuffer_));
    RT_CHECK(vx_mem_address(triExBuffer_, &kernel_arg_.triEx_addr));

    // allocate triIdx buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->triIdx_buf().size() * sizeof(uint32_t), VX_MEM_READ, &idxBuffer_));
    RT_CHECK(vx_mem_address(idxBuffer_, &kernel_arg_.triIdx_addr));

    // allocate tlas buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->tlas_nodes().size() * sizeof(tlas_node_t), VX_MEM_READ, &tlasBuffer_));
    RT_CHECK(vx_mem_address(tlasBuffer_, &kernel_arg_.tlas_addr));

    // allocate inst buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->blas_nodes().size() * sizeof(blas_node_t), VX_MEM_READ, &blasBuffer_));
    RT_CHECK(vx_mem_address(blasBuffer_, &kernel_arg_.blas_addr));

    // allocate bvh buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->bvh_nodes().size() * sizeof(bvh_node_t), VX_MEM_READ, &bvhBuffer_));
    RT_CHECK(vx_mem_address(bvhBuffer_, &kernel_arg_.bvh_addr));

    // allocate tex buffer
    RT_CHECK(vx_mem_alloc(device_, scene_->tex_buf().size(), VX_MEM_READ, &texBuffer_));
    RT_CHECK(vx_mem_address(texBuffer_, &kernel_arg_.tex_addr));

    // allocate output buffer
    RT_CHECK(vx_mem_alloc(device_, dst_width_ * dst_height_ * sizeof(uint32_t), VX_MEM_WRITE, &output_buffer_));
    RT_CHECK(vx_mem_address(output_buffer_, &kernel_arg_.dst_addr));
  }

  return 0;
}

int Tracer::setup(float camera_vfov, float zoom, float3_t light_pos, float3_t light_color, float3_t ambient_color, float3_t background_color) {

  // transform BVH instances
  {
    // slightly rotate the scene
    auto T = mat4_t::RotateX(-PI / 4) * mat4_t::RotateY(PI / 4);
    scene_->applyTransform(T);
  }

  // build the scene
  scene_->build();
  kernel_arg_.tlas_root = scene_->tlas_root();

  // setup Camera
  {
    float3_t camera_pos, camera_target, camera_up;
    scene_->computeFramingCamera(camera_vfov * DEG2RAD, zoom, &camera_pos, &camera_target, &camera_up);
    float3_t forward = normalize(camera_target - camera_pos);
    float3_t right = normalize(cross(forward, camera_up));
    float3_t up = cross(right, forward);
    kernel_arg_.camera_pos = camera_pos;
    kernel_arg_.camera_forward = forward;
    kernel_arg_.camera_right = right;
    kernel_arg_.camera_up = up;
  }

  // setup viewplane
  {
    float aspect_ratio = float(dst_width_) / dst_height_;
    float viewport_height = 2.0f * tan(camera_vfov * 0.5f);
    float viewport_width = viewport_height * aspect_ratio;
    kernel_arg_.viewplane = {viewport_width, viewport_height};
  }

  // setup lighting
  {
    kernel_arg_.light_pos = light_pos;
    kernel_arg_.light_color = light_color;
    kernel_arg_.ambient_color = ambient_color;
    kernel_arg_.background_color = background_color;
  }

  if (use_cpu_)
    return 0;

  // upload tri data
  RT_CHECK(vx_copy_to_dev(triBuffer_, scene_->tri_buf().data(), 0, scene_->tri_buf().size() * sizeof(tri_t)));

  // upload triEx data
  RT_CHECK(vx_copy_to_dev(triExBuffer_, scene_->triEx_buf().data(), 0, scene_->triEx_buf().size() * sizeof(tri_ex_t)));

  // upload triIdx data
  RT_CHECK(vx_copy_to_dev(idxBuffer_, scene_->triIdx_buf().data(), 0, scene_->triIdx_buf().size() * sizeof(uint32_t)));

  // upload tlas data
  RT_CHECK(vx_copy_to_dev(tlasBuffer_, scene_->tlas_nodes().data(), 0, scene_->tlas_nodes().size() * sizeof(tlas_node_t)));

  // upload inst data
  RT_CHECK(vx_copy_to_dev(blasBuffer_, scene_->blas_nodes().data(), 0, scene_->blas_nodes().size() * sizeof(blas_node_t)));

  // upload bvh data
  RT_CHECK(vx_copy_to_dev(bvhBuffer_, scene_->bvh_nodes().data(), 0, scene_->bvh_nodes().size() * sizeof(bvh_node_t)));

  // upload tex data
  RT_CHECK(vx_copy_to_dev(texBuffer_, scene_->tex_buf().data(), 0, scene_->tex_buf().size()));

  return 0;
}

int Tracer::run(const char *output_file) {
  std::vector<uint8_t> h_output(dst_width_ * dst_height_ * sizeof(uint32_t));

  std::cout << "Begin rendering to " << dst_width_ << "x" << dst_height_ << " framebuffer." << std::endl;

  if (use_cpu_) {
    kernel_arg_.dst_addr = (uint64_t)h_output.data();
    this->render();
  } else {
    // upload kernel arguments
    RT_CHECK(vx_upload_bytes(device_, &kernel_arg_, sizeof(kernel_arg_t), &args_buffer_));

    // start kernel execution
    RT_CHECK(vx_start(device_, krnl_buffer_, args_buffer_));

    // wait for the kernel to finish
    RT_CHECK(vx_ready_wait(device_, VX_MAX_TIMEOUT));

    // download the output buffer
    RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer_, 0, h_output.size()));
  }

  // write the output to a PPM file
  write_ppm(h_output.data(), dst_width_, dst_height_, output_file);

  return 0;
}

void Tracer::render() {
  auto arg = &kernel_arg_;
  auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);
  for (uint32_t y = 0; y < arg->dst_height; ++y) {
    for (uint32_t x = 0; x < arg->dst_width; ++x) {
      uint32_t out_idx = y * arg->dst_width + x;
      float3_t color = float3_t(0, 0, 0);
      for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
        auto ray = GenerateRay(x, y, arg);
        color += Trace(ray, arg);
      }
      out_ptr[out_idx] = RGB32FtoRGB8(color);
    }
  }
}