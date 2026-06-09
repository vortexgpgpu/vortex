#pragma once

#include "bvh.h"
#include "mesh.h"

// Manages all meshes in the scene
class Scene {
public:
  Scene(const std::vector<Mesh*> &meshes);
  ~Scene();

  int init();

  float computeFramingVfov(const float3_t &camera_pos, const float3_t &camera_target, const float3_t &camera_up, float aspect_ratio) const;

  void computeFramingCamera(float vfov, float zoon, float3_t *camera_pos, float3_t *camera_target, float3_t *camera_up);

  void applyTransform(const mat4_t &transform);

  void build();

  const auto &tlas_nodes() const { return tlas_->nodes(); }
  uint32_t tlas_root() const { return tlas_->rootIndex(); }

  const auto &blas_nodes() const { return blas_nodes_; }
  const auto &bvh_nodes() const { return bvh_nodes_; }

  const auto &tri_buf() const { return tri_buf_; }
  const auto &triEx_buf() const { return triEx_buf_; }
  const auto &triIdx_buf() const { return triIdx_buf_; }
  const auto &tex_buf() const { return tex_buf_; }

private:

  void arrangeMeshesAroundY(float margin);

  std::vector<Mesh *> meshes_;
  std::vector<BVH *> bvh_list_;
  std::vector<float3_t> centroids_;
  TLAS *tlas_;

  std::vector<blas_node_t> blas_nodes_;
  std::vector<bvh_node_t> bvh_nodes_;
  std::vector<tri_t> tri_buf_;
  std::vector<tri_ex_t> triEx_buf_;
  std::vector<uint32_t> triIdx_buf_;
  std::vector<uint8_t> tex_buf_;
};
