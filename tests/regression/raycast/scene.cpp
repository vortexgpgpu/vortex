#include "scene.h"

Scene::Scene(const std::vector<Mesh*> &meshes)
  : meshes_(meshes), bvh_list_(meshes.size(), nullptr)
{}

Scene::~Scene() {
  for (auto mesh : meshes_) {
    delete mesh;
  }
  for (auto bvh : bvh_list_) {
    delete bvh;
  }
  delete tlas_;
}

int Scene::init() {
  // estimate buffers size
  uint32_t num_tris = 0;
  uint64_t texture_size = 0;
  for (auto mesh : meshes_) {
    num_tris += mesh->tri().size();
    texture_size += mesh->texture()->size();
  }

  // allocate buffers
  tex_buf_.resize(texture_size);
  tri_buf_.resize(num_tris);
  triEx_buf_.resize(num_tris);
  triIdx_buf_.resize(num_tris);
  centroids_.resize(num_tris);
  bvh_nodes_.resize(num_tris * 2);
  blas_nodes_.resize(meshes_.size());

  // create BVH objects
  uint32_t bvh_offset = 0;
  uint32_t tri_offset = 0;
  uint64_t tex_offset = 0;

  for (uint32_t i = 0; i < meshes_.size(); ++i) {
    auto mesh = meshes_.at(i);

    // copy mesh buffers
    memcpy(tri_buf_.data() + tri_offset, mesh->tri().data(), mesh->tri().size() * sizeof(tri_t));
    memcpy(triEx_buf_.data() + tri_offset, mesh->triEx().data(), mesh->tri().size() * sizeof(tri_ex_t));
    memcpy(tex_buf_.data() + tex_offset, mesh->texture()->pixels(), mesh->texture()->size());

    // precompute triangle indices & centroids
    for (uint32_t j = 0; j < mesh->tri().size(); ++j) {
      auto &tri = mesh->tri().at(j);
      triIdx_buf_[tri_offset + j] = tri_offset + j;
      centroids_[tri_offset + j] = (tri.v0 + tri.v1 + tri.v2) / 3;
    }

    // setup blas node
    auto &blas_node = blas_nodes_.at(i);
    blas_node.bvh_offset = bvh_offset;
    blas_node.tex_offset = tex_offset;
    blas_node.transform = mat4_t::Identity();
    blas_node.invTransform = mat4_t::Identity();
    blas_node.tex_width = mesh->texture()->width();
    blas_node.tex_height = mesh->texture()->height();
    blas_node.reflectivity = mesh->reflectivity();
    // create BVH

    auto bvh = new BVH(tri_buf_.data(), centroids_.data(), mesh->tri().size(), bvh_nodes_.data() + bvh_offset, triIdx_buf_.data() + tri_offset);

    // update offsets
    bvh_offset += bvh->nodeCount();
    tri_offset += mesh->tri().size();
    tex_offset += mesh->texture()->size();
    bvh_list_[i] = bvh;
  }

  // create TLAS
  tlas_ = new TLAS(bvh_list_, blas_nodes_.data());

  // position meshes around Y axis
  this->arrangeMeshesAroundY(0.0f);

  return 0;
}

float Scene::computeFramingVfov(const float3_t &camera_pos,
                                const float3_t &camera_target,
                                const float3_t &camera_up,
                                float aspect_ratio) const {
  // Compute camera basis vectors
  float3_t forward = normalize(camera_target - camera_pos);
  float3_t right = normalize(cross(forward, camera_up));
  float3_t up = normalize(cross(right, forward));

  float max_half_angle_y = 0.0f;
  float max_half_angle_x = 0.0f;

  for (auto &node : tlas_->nodes()) {
    float3_t corners[8] = {
        float3_t(node.aabbMin.x, node.aabbMin.y, node.aabbMin.z),
        float3_t(node.aabbMax.x, node.aabbMin.y, node.aabbMin.z),
        float3_t(node.aabbMin.x, node.aabbMax.y, node.aabbMin.z),
        float3_t(node.aabbMin.x, node.aabbMin.y, node.aabbMax.z),
        float3_t(node.aabbMax.x, node.aabbMax.y, node.aabbMin.z),
        float3_t(node.aabbMax.x, node.aabbMin.y, node.aabbMax.z),
        float3_t(node.aabbMin.x, node.aabbMax.y, node.aabbMax.z),
        float3_t(node.aabbMax.x, node.aabbMax.y, node.aabbMax.z)};

    for (const auto &corner : corners) {
      float3_t dir = corner - camera_pos;
      float dist_along_forward = dot(dir, forward) * 2;

      // Skip points behind or exactly at the camera plane
      if (dist_along_forward <= 0.0f)
        continue;

      // Calculate projection distances
      float dist_right = dot(dir, right);
      float dist_up = dot(dir, up);

      // Calculate angles
      float half_angle_x = atan2(dist_right, dist_along_forward);
      float half_angle_y = atan2(dist_up, dist_along_forward);

       // Update maxima
      max_half_angle_y = std::max(max_half_angle_y, std::abs(half_angle_y));
      max_half_angle_x = std::max(max_half_angle_x, std::abs(half_angle_x));
    }
  }

  // Calculate required FOVs
  const float required_vfov = 2.0f * max_half_angle_y;
  const float required_hfov = 2.0f * max_half_angle_x;
  const float hfov_based_vfov = required_hfov / aspect_ratio;

  // Return the larger FOV that covers all cases
  return std::max(required_vfov, hfov_based_vfov);
}

void Scene::computeFramingCamera(float vfov,
                                 float zoon,
                                 float3_t *camera_pos,
                                 float3_t *camera_target,
                                 float3_t *camera_up) {
  // 1) Compute scene AABB
  float3_t bmin = { +LARGE_FLOAT, +LARGE_FLOAT, +LARGE_FLOAT };
  float3_t bmax = { -LARGE_FLOAT, -LARGE_FLOAT, -LARGE_FLOAT };
  for (auto &node : tlas_->nodes()) {
    bmin.x = std::min(bmin.x, node.aabbMin.x);
    bmin.y = std::min(bmin.y, node.aabbMin.y);
    bmin.z = std::min(bmin.z, node.aabbMin.z);
    bmax.x = std::max(bmax.x, node.aabbMax.x);
    bmax.y = std::max(bmax.y, node.aabbMax.y);
    bmax.z = std::max(bmax.z, node.aabbMax.z);
  }

  // 2) Get bounding sphere (center + radius)
  float3_t center = (bmin + bmax) * 0.5f;
  float radius = length(bmax - center);

  // 3) Set target and up
  *camera_target = center;
  *camera_up = float3_t{0.0f, 1.0f, 0.0f};

  // 4) Compute how far back to position the camera
  float distance = radius / std::tan(vfov);

  // 5) Apply zoom factor (<1 = closer, >1 = farther)
  distance *= zoon;

  // 6) Dolly along -Z so that the scene is framed
  //    (we assume "forward" = +Z in object space, so camera sits at -Z)
  *camera_pos = center - float3_t{0.0f, 0.0f, 1.0f} * distance;
}

void Scene::arrangeMeshesAroundY(float margin) {
  if (meshes_.size() == 1)
    return;

  uint32_t N = meshes_.size();

  // 1. Calculate bounding sizes
  std::vector<float> radii(meshes_.size());
  for (size_t i = 0; i < N; i++) {
    auto bvh = bvh_list_[i];
    auto &bmin = bvh->aabbMin();
    auto &bmax = bvh->aabbMax();
    float dx = bmax.x - bmin.x;
    float dz = bmax.z - bmin.z;
    float radius = 0.5f * std::sqrt(dx*dx + dz*dz);
    radii[i] = radius + margin;
  }

  // 2. Find the maximum sum of two adjacent R's
  float maxPairSum = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    size_t j = (i + 1) % N;
    maxPairSum = std::max(maxPairSum, radii[i] + radii[j]);
  }

  // 3. Compute the needed circle radius so that 2*R*sin(pi/N) >= maxPairSum
  float angleStep = 2.0f * M_PI / float(N);
  float sinHalfStep = std::sin(angleStep / 2.0f);
  float arrangementR = maxPairSum / (2.0f * sinHalfStep);

  // 4. Position each mesh
  for (size_t i = 0; i < N; i++) {
    auto &node = blas_nodes_[i];
    float theta = angleStep * float(i);
    float x = arrangementR * std::cos(theta);
    float z = arrangementR * std::sin(theta);
    auto T = mat4_t::Translate(float3_t(x, 0.0f, z));
    node.applyTransform(T);
  }
}

void Scene::applyTransform(const mat4_t &transform) {
  for (auto &node : blas_nodes_) {
    node.applyTransform(transform);
  }
}

void Scene::build() {
  // build TLAS
  tlas_->build();
}