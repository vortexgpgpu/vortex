#include "bvh.h"
#include "kdtree.h"
#include <utility>

// bin count for binned BVH building
#define BINS 8

// BVH class implementation

BVH::BVH(const tri_t *triData, const float3_t *centroids, uint32_t triCount, bvh_node_t *bvh_nodes, uint32_t *triIndices) {
  bvhNodes_ = bvh_nodes;
  centroids_ = centroids;
  triCount_ = triCount;
  triData_ = triData;
  triIndices_ = triIndices;
  this->build();
}

BVH::~BVH() {
  //--
}

void BVH::build() {
  // Recursive build staring at the root node
  bvh_node_t &root = bvhNodes_[nodeCount_++];
  this->initializeNode(root, 0, triCount_);
}

void BVH::initializeNode(bvh_node_t &node, uint32_t first, uint32_t count) {
  node.leftFirst = first;
  node.triCount = count;

  float3_t centroidMin, centroidMax;
  this->updateNodeBounds(node, &centroidMin, &centroidMax);
  if (count > 1) {
    this->subdivide(node, centroidMin, centroidMax);
  }
}

void BVH::subdivide(bvh_node_t &node, const float3_t &centroidMin, const float3_t &centroidMax) {
  // determine split axis using Surface Area Heuristic (SAH)
  uint32_t axis, splitPos;
  float splitCost = findBestSplitPlane(node, centroidMin, centroidMax, &axis, &splitPos);
  float nosplitCost = node.calculateNodeCost();
  if (splitCost >= nosplitCost)
    return;

  // partition triangles
  uint32_t leftCount = partitionTriangles(node, axis, splitPos, centroidMin, centroidMax);
  if (leftCount == 0 || leftCount == node.triCount)
    return;

  // create child nodes
  uint32_t leftChildIdx = nodeCount_++;
  uint32_t rightChildIdx = nodeCount_++;

  auto &leftChild = bvhNodes_[leftChildIdx];
  auto &rightChild = bvhNodes_[rightChildIdx];

  this->initializeNode(leftChild, node.leftFirst, leftCount);
  this->initializeNode(rightChild, node.leftFirst + leftCount, node.triCount - leftCount);

  // update parent's child nodes
  node.leftFirst = leftChildIdx;
  node.triCount = 0; // mark as parent node
}

uint32_t BVH::partitionTriangles(const bvh_node_t &node, uint32_t axis, uint32_t splitPos, const float3_t &centroidMin, const float3_t &centroidMax) const {
  float scale = BINS / (centroidMax[axis] - centroidMin[axis]);
  uint32_t *triPtr = triIndices_ + node.leftFirst;

  uint32_t i = 0;
  uint32_t j = node.triCount - 1;

  while (i <= j) {
    uint32_t triIdx = triPtr[i];
    auto &centroid = centroids_[triIdx];
    uint32_t bin = clamp(int((centroid[axis] - centroidMin[axis]) * scale), 0, BINS - 1);
    if (bin < splitPos) {
      i++;
    } else {
      std::swap(triPtr[i], triPtr[j--]);
    }
  }

  return i;
}

float BVH::findBestSplitPlane(const bvh_node_t &node, const float3_t &centroidMin, const float3_t &centroidMax, uint32_t *axis, uint32_t *splitPos) const {
  float bestCost = LARGE_FLOAT;
  for (uint32_t a = 0; a < 3; a++) {
    float boundsMin = centroidMin[a], boundsMax = centroidMax[a];
    if (boundsMin == boundsMax)
      continue;

    // populate the bins
    float scale = BINS / (boundsMax - boundsMin);
    float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
    int leftSum = 0, rightSum = 0;

    struct Bin {
      AABB bounds;
      int triCount = 0;
    } bin[BINS];

    for (uint32_t i = 0; i < node.triCount; i++) {
      auto triIdx = triIndices_[node.leftFirst + i];
      auto &triangle = triData_[triIdx];
      auto &centroid = centroids_[triIdx];
      int binIdx = std::min(BINS - 1, (int)((centroid[a] - boundsMin) * scale));
      bin[binIdx].triCount++;
      bin[binIdx].bounds.grow(triangle.v0);
      bin[binIdx].bounds.grow(triangle.v1);
      bin[binIdx].bounds.grow(triangle.v2);
    }

    // gather data for the 7 planes between the 8 bins
    AABB leftBox, rightBox;
    for (int i = 0; i < BINS - 1; i++) {
      leftSum += bin[i].triCount;
      leftBox.grow(bin[i].bounds);
      leftCountArea[i] = leftSum * leftBox.area();
      rightSum += bin[BINS - 1 - i].triCount;
      rightBox.grow(bin[BINS - 1 - i].bounds);
      rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
    }

    // calculate SAH cost for the 7 planes
    scale = (boundsMax - boundsMin) / BINS;
    for (int i = 0; i < BINS - 1; i++) {
      const float planeCost = leftCountArea[i] + rightCountArea[i];
      if (planeCost < bestCost) {
        *axis = a;
        *splitPos = i + 1;
        bestCost = planeCost;
      }
    }
  }
  return bestCost;
}

void BVH::updateNodeBounds(bvh_node_t &node, float3_t *centroidMin, float3_t *centroidMax) const {
  node.aabbMin = float3_t(LARGE_FLOAT);
  node.aabbMax = float3_t(-LARGE_FLOAT);
  auto centroid_min = float3_t(LARGE_FLOAT);
  auto centroid_max = float3_t(-LARGE_FLOAT);
  for (uint32_t first = node.leftFirst, i = 0; i < node.triCount; i++) {
    uint32_t triIdx = triIndices_[first + i];
    auto &tri = triData_[triIdx];
    node.aabbMin = fminf(node.aabbMin, tri.v0);
    node.aabbMin = fminf(node.aabbMin, tri.v1);
    node.aabbMin = fminf(node.aabbMin, tri.v2);
    node.aabbMax = fmaxf(node.aabbMax, tri.v0);
    node.aabbMax = fmaxf(node.aabbMax, tri.v1);
    node.aabbMax = fmaxf(node.aabbMax, tri.v2);
    auto &centroid = centroids_[triIdx];
    centroid_min = fminf(centroid_min, centroid);
    centroid_max = fmaxf(centroid_max, centroid);
  }
  *centroidMin = centroid_min;
  *centroidMax = centroid_max;
}

// TLAS implementation

TLAS::TLAS(const std::vector<BVH *> &bvh_list, const blas_node_t *blas_nodes)
    : bvh_list_(bvh_list) {
  blas_nodes_ = blas_nodes;
  blasCount_ = bvh_list.size();
  nodeCount_ = 2 * blasCount_ - 1;
  // allocate TLAS nodes
  tlasNodes_.resize(nodeCount_);
  nodeIndices_.resize(blasCount_);
  triCounts_.resize(blasCount_);
}

TLAS::~TLAS() {
  //--
}

void TLAS::build() {
  if (blasCount_ == 0)
    return;

  // Initialize leaf nodes
  for (uint32_t i = 0; i < blasCount_; ++i) {
    auto &bvh = bvh_list_[i];
    auto &blas_node = blas_nodes_[i];

    // calculate world-space bounds using the new matrix
    auto &aabbMin = bvh->aabbMin();
    auto &aabbMax = bvh->aabbMax();
    AABB bounds;
    for (int c = 0; c < 8; ++c) {
      float3_t pos(c & 1 ? aabbMax.x : aabbMin.x,
                   c & 2 ? aabbMax.y : aabbMin.y,
                   c & 4 ? aabbMax.z : aabbMin.z);
      bounds.grow(TransformPosition(pos, blas_node.transform));
    }

    tlasNodes_[i].aabbMin = bounds.bmin;
    tlasNodes_[i].aabbMax = bounds.bmax;
    tlasNodes_[i].blasIdx = i;
    tlasNodes_[i].setLeftRight(0, 0); // leaf node

    triCounts_[i] = bvh->triCount();
    nodeIndices_[i] = i;
  }

  uint32_t currentInternalNodeIndex = blasCount_;
  rootIndex_ = buildRecursive(0, blasCount_ - 1, currentInternalNodeIndex);
}

uint32_t TLAS::buildRecursive(uint32_t start, uint32_t end, uint32_t &currentInternalNodeIndex) {
  if (start == end) {
    return nodeIndices_[start]; // Leaf node
  }

  // Compute current AABB
  float3_t aabbMin = tlasNodes_[nodeIndices_[start]].aabbMin;
  float3_t aabbMax = tlasNodes_[nodeIndices_[start]].aabbMax;
  for (uint32_t i = start + 1; i <= end; ++i) {
    auto &node = tlasNodes_[nodeIndices_[i]];
    aabbMin = fminf(aabbMin, node.aabbMin);
    aabbMax = fmaxf(aabbMax, node.aabbMax);
  }

  // Determine best split axis and position using SAH
  int splitAxis = 0;
  float splitPos = 0.0f;
  float bestCost = std::numeric_limits<float>::infinity();

  float3_t extent = aabbMax - aabbMin;
  for (uint32_t axis = 0; axis < 3; ++axis) {
    if (extent[axis] <= 0)
      continue;

    float binWidth = extent[axis] / BINS;
    for (uint32_t i = 1; i < BINS; ++i) {
      float candidatePos = aabbMin[axis] + i * binWidth;

      // Calculate SAH cost
      uint32_t leftTris = 0, rightTris = 0;
      float3_t leftMin(LARGE_FLOAT), leftMax(-LARGE_FLOAT);
      float3_t rightMin(LARGE_FLOAT), rightMax(-LARGE_FLOAT);

      for (uint32_t j = start; j <= end; ++j) {
        const tlas_node_t &node = tlasNodes_[nodeIndices_[j]];
        float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
        if (centroid < candidatePos) {
          leftMin = fminf(leftMin, node.aabbMin);
          leftMax = fmaxf(leftMax, node.aabbMax);
          leftTris += triCounts_[node.blasIdx];
        } else {
          rightMin = fminf(rightMin, node.aabbMin);
          rightMax = fmaxf(rightMax, node.aabbMax);
          rightTris += triCounts_[node.blasIdx];
        }
      }
      if (leftTris == 0 || rightTris == 0)
        continue; // no valid split

      // Compute SAH cost
      float leftArea = surfaceArea(leftMin, leftMax);
      float rightArea = surfaceArea(rightMin, rightMax);
      float cost = leftArea * leftTris + rightArea * rightTris;
      if (cost < bestCost) {
        bestCost = cost;
        splitAxis = axis;
        splitPos = candidatePos;
      }
    }
  }

  // Fallback to median split if SAH failed
  if (bestCost == std::numeric_limits<float>::infinity()) {
    splitAxis = (extent.x > extent.y) ? ((extent.x > extent.z) ? 0 : 2) : ((extent.y > extent.z) ? 1 : 2);
    // Compute median centroid along splitAxis
    std::vector<float> centroids;
    for (uint32_t i = start; i <= end; ++i) {
        const auto &node = tlasNodes_[nodeIndices_[i]];
        float centroid = (node.aabbMin[splitAxis] + node.aabbMax[splitAxis]) * 0.5f;
        centroids.push_back(centroid);
    }
    std::sort(centroids.begin(), centroids.end());
    splitPos = centroids[centroids.size() / 2]; // median
  }

  // Partition the primitives based on the best split
  uint32_t mid = partition(start, end, splitAxis, splitPos);
  if (mid == start || mid == end) {
    mid = (start + end) / 2;
  }

  // Recursively build left and right subtrees
  uint32_t leftChild = buildRecursive(start, mid, currentInternalNodeIndex);
  uint32_t rightChild = buildRecursive(mid + 1, end, currentInternalNodeIndex);

  // Create internal node
  uint32_t nodeIndex = currentInternalNodeIndex++;
  auto &node = tlasNodes_[nodeIndex];
  node.setLeftRight(leftChild, rightChild);
  node.aabbMin = aabbMin;
  node.aabbMax = aabbMax;

  return nodeIndex;
}

uint32_t TLAS::partition(int start, int end, int axis, float splitPos) {
  int left = start;
  int right = end;

  while (left <= right) {
    while (left <= end) {
      auto &node = tlasNodes_[nodeIndices_[left]];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid < splitPos)
        left++;
      else
        break;
    }
    while (right >= start) {
      auto &node = tlasNodes_[nodeIndices_[right]];
      float centroid = (node.aabbMin[axis] + node.aabbMax[axis]) / 2;
      if (centroid >= splitPos)
        right--;
      else
        break;
    }
    if (left < right) {
      std::swap(nodeIndices_[left], nodeIndices_[right]);
      left++;
      right--;
    }
  }

  // All elements < splitPos → force split at last element
  if (right < start)
    return end;

  // All elements >= splitPos → force split at first element
  if (left > end)
    return start;

  // Return partition point
  return right;
}