#pragma once

#include "geometry.h"

// header for AVX, and every technology before it.
// if your CPU does not support this (unlikely), include the appropriate header instead.
// see: https://stackoverflow.com/a/11228864/2844473
#include <immintrin.h>

// custom kD-tree, used for quick TLAS construction
class KDTree {
public:
  struct KDNode {
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
    union {
      struct {
        uint left, right, parax;
        float splitPos;
      }; // for an interior node
      struct {
        uint first, count, dummy1, dummy2;
      }; // for a leaf node, 16 bytes
    };
    union {
      __m128 bmin4;
      float3_t bmin;
    }; // 16 bytes
    union {
      __m128 bmax4;
      float3_t bmax;
    }; // 16 bytes
    union {
      __m128 minSize4;
      float3_t minSize;
    }; // 16 bytes, total: 64 bytes
  #pragma GCC diagnostic pop
    bool isLeaf() { return (parax & 7) > 3; }
  };
  void swap(const uint a, const uint b) {
    uint t = tlasIdx[a];
    tlasIdx[a] = tlasIdx[b];
    tlasIdx[b] = t;
  }
  KDTree() = default;
  KDTree(tlas_node_t *tlasNodes, const uint N, const uint O) {
    // allocate space for nodes and indices
    tlas = tlasNodes; // copy of the original array of tlas nodes
    blasCount = N;    // blasCount remains constant
    tlasCount = N;    // tlasCount will grow during aggl. clustering
    offset = O;       // index of the first TLAS node in the array
    if (!leaf)
      leaf = new uint[100000];
    node = (KDNode *)std::aligned_alloc(64, N * 2 * sizeof(KDNode)); // pre-allocate kdtree nodes, aligned
    tlasIdx = new uint[N * 2];                                       // tlas array indirection so we can store ranges of nodes in leaves
  }
  void rebuild() {
    // we'll assume we get the same number of TLAS nodes each time
    tlasCount = blasCount;
    for (uint i = 0; i < blasCount; i++)
      tlasIdx[i] = i;
    // subdivide root node
    node[0].first = 0, node[0].count = blasCount, node[0].parax = 7;
    nodePtr = 1;        // root = 0, so node 1 is the first node we can create
    subdivide(node[0]); // recursively subdivide the root node
    // "each node keeps it's cluster's minimum box sizes in each axis"
    for (int i = nodePtr - 1; i >= 0; i--)
      if (node[i].isLeaf()) {
        node[i].minSize = float3_t(LARGE_FLOAT);
        for (uint j = 0; j < node[i].count; j++) {
          uint idx = tlasIdx[node[i].first + j];
          leaf[idx + offset] = i; // we can find tlas[idx] in leaf node[i]
          float3_t tlSize = 0.5f * (tlas[idx].aabbMax - tlas[idx].aabbMin);
          node[i].minSize = fminf(node[i].minSize, tlSize);
        }
      } else
        node[i].minSize = fminf(node[node[i].left].minSize, node[node[i].right].minSize);
  }
  void recurseRefit(uint idx) {
    while (1) {
      if (idx == 0)
        break;
      idx = node[idx].parax >> 3;
      node[idx].minSize = fminf(node[node[idx].left].minSize, node[node[idx].right].minSize);
      node[idx].bmin = fminf(node[node[idx].left].bmin, node[node[idx].right].bmin);
      node[idx].bmax = fmaxf(node[node[idx].left].bmax, node[node[idx].right].bmax);
    }
  }
  void subdivide(KDNode &node, uint depth = 0) {
    // update node bounds
    node.bmin = float3_t(LARGE_FLOAT), node.bmax = float3_t(-LARGE_FLOAT);
    node.minSize = float3_t(LARGE_FLOAT);
    for (uint i = 0; i < node.count; i++) {
      tlas_node_t &tln = tlas[tlasIdx[node.first + i]];
      float3_t C = (tln.aabbMin + tln.aabbMax) * 0.5f;
      node.minSize = fminf(node.minSize, 0.5f * (tln.aabbMax - tln.aabbMin));
      node.bmin = fminf(node.bmin, C), node.bmax = fmaxf(node.bmax, C);
    }
    // terminate if we are down to 1 tlas
    if (node.count < 2)
      return;
    // claim left and right child nodes
    uint axis = dominantAxis(node.bmax - node.bmin);
    float center = (node.bmin[axis] + node.bmax[axis]) * 0.5f;
#if 1
    // try to balance (works quite well but doesn't seem to pay off)
    if (node.count > 150) {
      // count how many would go to the left
      int leftCount = 0;
      for (uint i = 0; i < node.count; i++) {
        tlas_node_t &tl = tlas[tlasIdx[node.first + i]];
        float3_t P = (tl.aabbMin + tl.aabbMax) * 0.5f;
        if (P[axis] <= center)
          leftCount++;
      }
      float ratio = std::max(0.15f, std::min(0.85f, (float)leftCount / (float)node.count));
      center = ratio * node.bmin[axis] + (1 - ratio) * node.bmax[axis];
    }
#endif
    partition(node, center, axis);
    if (this->node[nodePtr].count == 0 || this->node[nodePtr + 1].count == 0)
      return; // split failed
    uint leftIdx = nodePtr;
    node.left = leftIdx, node.right = leftIdx + 1, nodePtr += 2;
    node.parax = (node.parax & 0xfffffff8) + axis, node.splitPos = center;
    subdivide(this->node[leftIdx], depth + 1);
    subdivide(this->node[leftIdx + 1], depth + 1);
  }
  void partition(KDNode &node, float splitPos, uint axis) {
    int N = node.count, first = node.first, last = first + N;
    if (N < 3)
      last = first + 1;
    else
      while (1) {
        tlas_node_t &tl = tlas[tlasIdx[first]];
        float3_t P = (tl.aabbMin + tl.aabbMax) * 0.5f;
        if (P[axis] > splitPos)
          swap(first, --last);
        else
          first++;
        if (first >= last)
          break;
      }
    KDNode &left = this->node[nodePtr];
    KDNode &right = this->node[nodePtr + 1];
    left.first = node.first, right.first = last;
    left.count = right.first - left.first;
    left.parax = right.parax = (((uint32_t)(&node - this->node)) << 3) + 7;
    right.count = N - left.count;
  }
  void add(uint idx) {
    // capture bounds of new node
    idx -= offset;
    // create an index for the new node
    tlas_node_t &newTLAS = tlas[idx];
    float3_t C = (newTLAS.aabbMin + newTLAS.aabbMax) * 0.5f;
    tlasIdx[tlasCount++] = idx;
    // claim a new KDNode for the tlas and make it a leaf
    uint leafIdx, intIdx, nidx;
    KDNode &leafNode = node[leafIdx = freed[0]];
    leaf[idx + offset] = leafIdx;
    leafNode.first = tlasCount - 1, leafNode.count = 1;
    leafNode.bmin = leafNode.bmax = C;
    leafNode.minSize = 0.5f * (newTLAS.aabbMax - newTLAS.aabbMin);
    // we'll also need a new interior node
    intIdx = freed[1];
    // see where we should insert it
    float3_t P = (newTLAS.aabbMin + newTLAS.aabbMax) * 0.5f;
    KDNode *n = &node[nidx = 0];
    while (1)
      if (n->isLeaf()) {
        float3_t Pn;
        if (nidx == 0) // special case: root is leaf ==> tree consists of only one node
        {
          node[intIdx] = node[0];  // interior node slot is overwritten with old root (now sibling)
          node[intIdx].parax &= 7; // sibling's parent is the root node
          node[leafIdx].parax = 7; // new node's parent is the root node
          // 'split' the new KDNode over the greatest axis of separation
          Pn = (node[intIdx].bmin + node[intIdx].bmax) * 0.5f;
          // and finally, redirect leaf entries for old root
          for (uint j = 0; j < node[intIdx].count; j++)
            leaf[tlasIdx[node[intIdx].first + j] + offset] = intIdx;
          // put the new leaf and n in the correct fields
          nidx = intIdx, intIdx = 0, node[intIdx].parax = 0;
        } else {
          // replace one child of the parent by the new interior node
          KDNode &parent = node[n->parax >> 3];
          if (parent.left == nidx)
            parent.left = intIdx;
          else
            parent.right = intIdx;
          // rewire parent pointers
          node[intIdx].parax = n->parax & 0xfffffff8;
          n->parax = leafNode.parax = (intIdx << 3) + 7;
          // 'split' the new KDNode over the greatest axis of separation
          Pn = (n->bmin + n->bmax) * 0.5f;
        }
        // put the new leaf and n in the correct fields
        uint axis = dominantAxis(P - Pn);
        node[intIdx].parax += axis;
        node[intIdx].splitPos = ((Pn + P) * 0.5f)[axis];
        if (P[axis] < node[intIdx].splitPos)
          node[intIdx].left = leafIdx, node[intIdx].right = nidx;
        else
          node[intIdx].right = leafIdx, node[intIdx].left = nidx;
        break;
      } else // traverse
        n = &node[nidx = ((P[n->parax & 7] < n->splitPos) ? n->left : n->right)];
    // refit
    recurseRefit(leaf[idx + offset]);
  }
  void removeLeaf(uint idx) {
    // determine which node to delete for tlas[idx]: must be a leaf
    uint toDelete = leaf[idx];
    idx -= offset;
    if (node[toDelete].count > 1) // special case: multiple TLASes in one node, rare
    {
      KDNode &n = node[toDelete];
      for (uint j = 0; j < n.count; j++)
        if (tlasIdx[n.first + j] == idx)
          tlasIdx[n.first + j] = tlasIdx[n.first + n.count-- - 1];
      freed[0] = nodePtr++, freed[1] = nodePtr++;
      return;
    }
    uint parentIdx = node[toDelete].parax >> 3;
    KDNode &parent = node[parentIdx];
    uint sibling = parent.left == toDelete ? parent.right : parent.left;
    node[sibling].parax = (parent.parax & 0xfffffff8) + (node[sibling].parax & 7);
    parent = node[sibling]; // by value, but rather elegant
    if (parent.isLeaf())    // redirect leaf entries if the sibling is a leaf
      for (uint j = 0; j < parent.count; j++)
        leaf[tlasIdx[parent.first + j] + offset] = parentIdx;
    else // make sure child nodes point to the new index
      node[parent.left].parax = (parentIdx << 3) + (node[parent.left].parax & 7),
      node[parent.right].parax = (parentIdx << 3) + (node[parent.right].parax & 7);
    freed[0] = sibling, freed[1] = toDelete;
  }
  int FindNearest(uint A, uint &startB, float &startSA) {
    // keep all hot data together
    A -= offset;
    alignas(64) struct TravState {
      __m128 Pa4, tlasAbmin4, tlasAbmax4;
      uint n, stackPtr, bestB;
      float smallestSA; // exactly one cacheline
    } state;
    uint stack[60];
    uint &n = state.n, &stackPtr = state.stackPtr, &bestB = state.bestB;
    float &smallestSA = state.smallestSA;
    n = 0, stackPtr = 0, smallestSA = startSA, bestB = startB - offset;
    // gather data for node A
    __m128 &tlasAbmin4 = state.tlasAbmin4;
    __m128 &tlasAbmax4 = state.tlasAbmax4;
    tlasAbmin4 = _mm_setr_ps(tlas[A].aabbMin.x, tlas[A].aabbMin.y, tlas[A].aabbMin.z, 0);
    tlasAbmax4 = _mm_setr_ps(tlas[A].aabbMax.x, tlas[A].aabbMin.y, tlas[A].aabbMin.z, 0);
    float3_t tlasAbmin = *(float3_t *)&state.tlasAbmin4;
    float3_t tlasAbmax = *(float3_t *)&state.tlasAbmax4;
    __m128 &Pa4 = state.Pa4;
    Pa4 = _mm_mul_ps(_mm_set_ps1(0.5f), _mm_add_ps(tlasAbmin4, tlasAbmax4));
    const __m128 half4 = _mm_set_ps1(0.5f);
    const __m128 extentA4 = _mm_sub_ps(tlasAbmax4, tlasAbmin4);
    const __m128 halfExtentA4 = _mm_mul_ps(half4, _mm_sub_ps(tlasAbmax4, tlasAbmin4));
    //const __m128 tmp4 = _mm_setr_ps(-1, -1, -1, 1);
    //const __m128 xyzMask4 = _mm_cmple_ps(tmp4, _mm_setzero_ps());
    // walk the tree
    while (1) {
      while (1) {
        if (node[n].isLeaf()) {
          // loop over the BLASes stored in this leaf
          for (uint i = 0; i < node[n].count; i++) {
            uint B = tlasIdx[node[n].first + i];
            if (B == A)
              continue;
            // calculate surface area of union of A and B
            const float3_t size = fmaxf(tlasAbmax, tlas[B].aabbMax) - fminf(tlasAbmin, tlas[B].aabbMin);
            const float SA = size.x * size.y + size.y * size.z + size.z * size.x;
            if (SA < smallestSA)
              smallestSA = SA, bestB = B;
          }
          break;
        }
        // consider recursing into branches, sorted by distance
        uint t, nearNode = node[n].left, farNode = node[n].right;
        alignas(16) float Pa4Array[4];
        _mm_store_ps(Pa4Array, Pa4);
        if (Pa4Array[node[n].parax & 7] > node[n].splitPos)
          t = nearNode, nearNode = farNode, farNode = t;

        const __m128 v0a = _mm_max_ps(_mm_sub_ps(node[nearNode].bmin4, Pa4), _mm_sub_ps(Pa4, node[nearNode].bmax4));
        const __m128 v0b = _mm_max_ps(_mm_sub_ps(node[farNode].bmin4, Pa4), _mm_sub_ps(Pa4, node[farNode].bmax4));

        alignas(16) float d4aArray[4], d4bArray[4];
        _mm_store_ps(d4aArray, _mm_max_ps(extentA4, _mm_sub_ps(v0a, _mm_add_ps(node[nearNode].minSize4, halfExtentA4))));
        _mm_store_ps(d4bArray, _mm_max_ps(extentA4, _mm_sub_ps(v0b, _mm_add_ps(node[farNode].minSize4, halfExtentA4))));

        const float sa1 = d4aArray[0] * d4aArray[1] + d4aArray[1] * d4aArray[2] + d4aArray[2] * d4aArray[0];
        const float sa2 = d4bArray[0] * d4bArray[1] + d4bArray[1] * d4bArray[2] + d4bArray[2] * d4bArray[0];

        const float diff1 = sa1 - smallestSA, diff2 = sa2 - smallestSA;
        const uint visit = (diff1 < 0) * 2 + (diff2 < 0);
        if (!visit)
          break;
        if (visit == 3)
          stack[stackPtr++] = farNode, n = nearNode;
        else if (visit == 2)
          n = nearNode;
        else
          n = farNode;
      }
      if (stackPtr == 0)
        break;
      n = stack[--stackPtr];
    }
    // all done; return best match
    startB = bestB + offset;
    startSA = smallestSA;
    return bestB + offset;
  }
  // data
  KDNode *node = 0;
  tlas_node_t *tlas = 0;
  uint *tlasIdx = 0, nodePtr = 1, tlasCount = 0, blasCount = 0, offset = 0, freed[2] = {0, 0};
  inline static uint *leaf = 0; // will be shared between trees
};