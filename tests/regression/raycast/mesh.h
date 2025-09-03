#pragma once

#include "bvh.h"
#include "surface.h"

struct obj_mesh_t {
  struct vert_t {
    uint32_t p;
    uint32_t n;
    uint32_t t;
  };

  struct face_t {
    uint32_t v[3];
  };

  std::vector<float3_t> positions;
  std::vector<float3_t> normals;
  std::vector<float2_t> texcoords;
  std::vector<vert_t>   vertices;
  std::vector<face_t>   faces;
};

int load_obj(const char *objFile, obj_mesh_t &mesh);

// 3D object container
class Mesh {
public:
  Mesh(const char *objFile, const char *texFile, float reflectivity);
  ~Mesh();

  const std::vector<tri_t>& tri() const { return tri_; }
  const std::vector<tri_ex_t>& triEx() const { return triEx_; }
  const Surface* texture() const { return texture_; }
  float reflectivity() const { return reflectivity_; }

private:
  std::vector<tri_t> tri_;
  std::vector<tri_ex_t> triEx_;
  Surface *texture_ = nullptr;
  float reflectivity_ = 0.0f;
};
