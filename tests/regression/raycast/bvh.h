#pragma once

#include "common.h"

class KDTree;

// minimalist AABB struct with grow functionality
struct AABB {
  float3_t bmin = LARGE_FLOAT;
  float3_t bmax = -LARGE_FLOAT;

  void grow(float3_t p) {
    bmin = fminf(bmin, p);
    bmax = fmaxf(bmax, p);
  }

  void grow(AABB &b) {
    if (b.bmin.x != LARGE_FLOAT) {
      grow(b.bmin);
      grow(b.bmax);
    }
  }

  float area() const {
    float3_t e = bmax - bmin; // box extent
    return e.x * e.y + e.y * e.z + e.z * e.x;
  }
};

// bounding volume hierarchy, to be used as BLAS
class BVH {
public:
  BVH(const tri_t *triData, const float3_t *centroids, uint32_t triCount, bvh_node_t* bvh_nodes, uint32_t *triIndices);
  ~BVH();

  auto &aabbMin() const { return bvhNodes_[0].aabbMin; }
  auto &aabbMax() const { return bvhNodes_[0].aabbMax; }

  bvh_node_t* nodes() const { return bvhNodes_; }
  uint32_t nodeCount() const { return nodeCount_; }

  uint32_t triCount() const { return triCount_; }
  const tri_t* triData() const { return triData_; }
  const uint32_t* triIndices() const { return triIndices_; }

private:

  void build();
  void initializeNode(bvh_node_t &node, uint32_t first, uint32_t count);
  void subdivide(bvh_node_t &node, const float3_t &centroidMin, const float3_t &centroidMax);
  void updateNodeBounds(bvh_node_t &node, float3_t *centroidMin, float3_t *centroidMax) const;
  uint32_t partitionTriangles(const bvh_node_t &node, uint32_t axis, uint32_t splitPos, const float3_t &centroidMin, const float3_t &centroidMax) const;
  float findBestSplitPlane(const bvh_node_t &node, const float3_t &centroidMin, const float3_t &centroidMax, uint32_t *axis, uint32_t *splitPos) const;

  uint32_t triCount_ = 0;        // number of triangles
  const tri_t *triData_ = nullptr; // pointer to mesh vertices
  const float3_t *centroids_ = nullptr; // triangle centroids
  uint32_t *triIndices_ = nullptr; // triangle indices
  bvh_node_t *bvhNodes_ = nullptr;
  uint32_t nodeCount_ = 0;
};

// top-level BVH class
class TLAS {
public:
  TLAS() = default;
  TLAS(const std::vector<BVH*>& bvh_list, const blas_node_t *blas_nodes);
  ~TLAS();

  void build();

  auto &nodes() const { return tlasNodes_; }

  uint32_t rootIndex() const { return rootIndex_; }

private:

  uint32_t buildRecursive(uint32_t start, uint32_t end, uint32_t &currentInternalNodeIndex);

  uint32_t partition(int start, int end, int axis, float splitPos);

  const std::vector<BVH*>& bvh_list_;
  const blas_node_t *blas_nodes_ = nullptr;
  std::vector<tlas_node_t> tlasNodes_;
  std::vector<uint32_t> nodeIndices_;
  std::vector<uint32_t> triCounts_;
  uint32_t blasCount_ = 0;
  uint32_t nodeCount_ = 0;
  uint32_t rootIndex_ = 0;
};

// EOF