#pragma once

#include "common.h"

#define BVH_STACK_SIZE 64

// Sample a texture using point filtering
float3_t texSample(const float2_t &uv, const uint32_t *pixels, uint32_t width, uint32_t height) {
  // Convert UVs to texel space
  uint32_t iu = uint32_t(uv.x * width);
  uint32_t iv = uint32_t(uv.y * height);

  // wrap coordinates
  iu %= width;
  iv %= height;

  // Sample texel
  uint32_t offset = (iu + iv * width);
  uint32_t texel = pixels[offset];
  return RGB8toRGB32F(texel);
}

// Sample a texture using bilinear filtering
float3_t texSampleBi(const float2_t &uv, const uint32_t *pixels, uint32_t width, uint32_t height) {
  // Convert UVs to texel space
  float u = uv.x * width;
  float v = uv.y * height;

  uint32_t x0 = (uint32_t)floorf(u);
  uint32_t y0 = (uint32_t)floorf(v);
  uint32_t x1 = x0 + 1;
  uint32_t y1 = y0 + 1;

  // Compute interpolation weights
  float fu = u - x0;
  float fv = v - y0;

  // wrap coordinates
  x0 %= width;
  y0 %= height;
  x1 %= width;
  y1 %= height;

  // Sample four texels
  float3_t c00 = RGB8toRGB32F(pixels[x0 + y0 * width]);
  float3_t c10 = RGB8toRGB32F(pixels[x1 + y0 * width]);
  float3_t c01 = RGB8toRGB32F(pixels[x0 + y1 * width]);
  float3_t c11 = RGB8toRGB32F(pixels[x1 + y1 * width]);

  // Interpolate horizontally
  float3_t cx0 = c00 * (1.0f - fu) + c10 * fu;
  float3_t cx1 = c01 * (1.0f - fu) + c11 * fu;

  // Interpolate vertically
  return cx0 * (1.0f - fv) + cx1 * fv;
}

float3_t diffuseLighting(const float3_t& pixel,
                         const float3_t& normal,
                         const float3_t& diffuse_color,
                         const float3_t& ambient_color,
                         const float3_t& light_color,
                         const float3_t& light_pos){
  float3_t L = light_pos - pixel;
  float dist = length(L);
  L *= 1.0f / dist;
  float att = 1.0f / (1.0f + dist * 0.1f);
  float NdotL = std::max(0.0f, dot(normal, L));
  return diffuse_color * (ambient_color + att * light_color * NdotL);
}


// Ray to BVH intersection test using closer-first traversal for early exit
void BVHIntersect(const ray_t &ray,
                  uint32_t blasIdx,
                  const bvh_node_t *bvhBuffer,
                  const uint32_t *triIdxBuffer,
                  const tri_t *triBuffer,
                  ray_hit_t *hit) {
  uint32_t stack[BVH_STACK_SIZE];
  uint32_t stackPtr = 0;
  stack[stackPtr++] = 0; // Push root node index

  while (stackPtr != 0) {
    uint32_t nodeIdx = stack[--stackPtr];
    const bvh_node_t &node = bvhBuffer[nodeIdx];
    if (node.isLeaf()) {
      // Intersect leaf triangles
      for (uint32_t i = 0; i < node.triCount; ++i) {
        uint32_t triIdx = triIdxBuffer[node.leftFirst + i];
        float dist;
        float3_t bcoords;
        if (ray.intersect(triBuffer[triIdx], &dist, &bcoords) && dist < hit->dist) {
          hit->dist = dist;
          hit->bcoords = bcoords;
          hit->blasIdx = blasIdx;
          hit->triIdx = triIdx;
        }
      }
    } else {
      // Process children
      uint32_t left  = node.leftFirst;
      uint32_t right = left + 1;

      float dLeft  = ray.intersect(bvhBuffer[left].aabbMin, bvhBuffer[left].aabbMax);
      float dRight = ray.intersect(bvhBuffer[right].aabbMin, bvhBuffer[right].aabbMax);

      // Early culling based on current hit distance
      bool hitLeft  = (dLeft != LARGE_FLOAT) && (dLeft < hit->dist);
      bool hitRight = (dRight != LARGE_FLOAT) && (dRight < hit->dist);

      // execute closer-first traversal
      if (hitLeft && hitRight) {
        if (dLeft < dRight) {
          std::swap(left, right);
        }
        stack[stackPtr++] = right;
        stack[stackPtr++] = left;
      } else if (hitLeft) {
        stack[stackPtr++] = left;
      } else if (hitRight) {
        stack[stackPtr++] = right;
      }
    }
  }
}

void BLASIntersect(const ray_t &ray,
                   uint32_t blasIdx,
                   const blas_node_t *blasBuffer,
                   const bvh_node_t *bvhBuffer,
                   const uint32_t *triIdxBuffer,
                   const tri_t *triBuffer,
                   ray_hit_t *hit) {
  auto &blas = blasBuffer[blasIdx];
  // backup and transform ray using instance transform
  ray_t backup = ray;
  backup.transform(blas.invTransform);
  // traverse the BLAS
  BVHIntersect(backup, blasIdx, bvhBuffer + blas.bvh_offset, triIdxBuffer, triBuffer, hit);
}

void TLASIntersect(const ray_t &ray,
                   uint32_t tlas_root,
                   const tlas_node_t *tlasBuffer,
                   const blas_node_t *blasBuffer,
                   const bvh_node_t *bvhBuffer,
                   const uint32_t *triIdxBuffer,
                   const tri_t *triBuffer,
                   ray_hit_t *hit) {
  uint32_t stack[BVH_STACK_SIZE];
  uint32_t stackPtr = 0;
  stack[stackPtr++] = tlas_root;

  while (stackPtr != 0) {
    uint32_t nodeIdx = stack[--stackPtr];
    const tlas_node_t &node = tlasBuffer[nodeIdx];

    if (node.isLeaf()) {
      // Intersect instance BLAS
      BLASIntersect(ray, node.blasIdx, blasBuffer, bvhBuffer, triIdxBuffer, triBuffer, hit);
    } else {
      // Process children
      uint32_t left = node.left();
      uint32_t right = node.right();

      float dLeft = ray.intersect(tlasBuffer[left].aabbMin, tlasBuffer[left].aabbMax);
      float dRight = ray.intersect(tlasBuffer[right].aabbMin, tlasBuffer[right].aabbMax);

      // Early culling based on current hit distance
      bool hitLeft = (dLeft != LARGE_FLOAT) && (dLeft < hit->dist);
      bool hitRight = (dRight != LARGE_FLOAT) && (dRight < hit->dist);

      // Execute closer-first traversal
      if (hitLeft && hitRight) {
        if (dLeft > dRight) {
          std::swap(left, right);
        }
        stack[stackPtr++] = right;
        stack[stackPtr++] = left;
      } else if (hitLeft) {
        stack[stackPtr++] = left;
      } else if (hitRight) {
        stack[stackPtr++] = right;
      }
    }
  }
}

ray_t GenerateRay(uint32_t x, uint32_t y, const kernel_arg_t *__UNIFORM__ arg) {
  // apply pixel center & convert to NDC [-0.5:0.5]
  float x_ndc = (x + 0.5f) / arg->dst_width - 0.5;
  float y_ndc = (y + 0.5f) / arg->dst_height - 0.5;

  // to viewplane space
  float x_vp = x_ndc * arg->viewplane.x;
  float y_vp = y_ndc * arg->viewplane.y;

  // to camera space
  auto pt_cam = x_vp * arg->camera_right + y_vp * arg->camera_up + arg->camera_forward;

  // to world space
  auto pt_w = pt_cam + arg->camera_pos;

  // construct ray
  auto camera_dir = normalize(pt_w - arg->camera_pos);
  return ray_t{arg->camera_pos, camera_dir};
}

float3_t Trace(const ray_t &ray, const kernel_arg_t *__UNIFORM__ arg) {
  auto tri_ptr = reinterpret_cast<const tri_t *>(arg->tri_addr);
  auto bvh_ptr = reinterpret_cast<const bvh_node_t *>(arg->bvh_addr);
  auto texIdx_ptr = reinterpret_cast<const uint32_t *>(arg->triIdx_addr);
  auto tlas_ptr = reinterpret_cast<const tlas_node_t *>(arg->tlas_addr);
  auto blas_ptr = reinterpret_cast<const blas_node_t *>(arg->blas_addr);
  auto triEx_ptr = reinterpret_cast<const tri_ex_t *>(arg->triEx_addr);
  auto tex_ptr = reinterpret_cast<const uint8_t *>(arg->tex_addr);

  ray_t cur_ray = ray;

  float3_t radiance = {0,0,0};
  float throughput = 1.0f;

  // bounce until we hit the background or a primitive
  for (uint32_t bounce = 0; bounce < arg->max_depth; ++bounce) {
    ray_hit_t hit;
    TLASIntersect(cur_ray, arg->tlas_root, tlas_ptr, blas_ptr, bvh_ptr, texIdx_ptr, tri_ptr, &hit);
    if (hit.dist == LARGE_FLOAT) {
      radiance += arg->background_color * throughput;
      break; // no hit!
    }

    // fetch instance & per-triangle data
    auto &blas = blas_ptr[hit.blasIdx];
    const tri_ex_t &triEx = triEx_ptr[hit.triIdx];

    // intersection point
    float3_t I = cur_ray.orig + cur_ray.dir * hit.dist;

    // interpolated, transformed normal
    float3_t N = triEx.N1 * hit.bcoords.x + triEx.N2 * hit.bcoords.y + triEx.N0 * hit.bcoords.z;
    mat4_t invTranspose = blas.invTransform.transposed();
    N = normalize(TransformVector(N, invTranspose));

    // barycentric UV
    float2_t uv = triEx.uv1 * hit.bcoords.x + triEx.uv2 * hit.bcoords.y + triEx.uv0 * hit.bcoords.z;

    // diffuse shading
    auto tex_pixels = reinterpret_cast<const uint32_t*>(tex_ptr + blas.tex_offset);
    float3_t texColor = texSample(uv, tex_pixels, blas.tex_width, blas.tex_height);
    float3_t diffuse = diffuseLighting(I, N, texColor, arg->ambient_color, arg->light_color, arg->light_pos);

    auto reflectivity = blas.reflectivity;

    // add non-reflected diffuse contribution
    radiance += throughput * diffuse * (1 - reflectivity);

    // carry forward reflected energy
    throughput *= reflectivity;

    // bounce if reflective
    if (reflectivity > 0.0f && bounce + 1 < arg->max_depth) {
      float3_t R = normalize(cur_ray.dir - 2.0f * N * dot(N, cur_ray.dir));
      cur_ray.orig = I + R * 0.001f;
      cur_ray.dir = R;
      continue;
    }

    // environment contribution for remaining throughput
    radiance += throughput * arg->background_color;

    break;
  }

  return radiance;
}