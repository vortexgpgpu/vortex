#include "mesh.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

// Mesh class implementation

std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int load_obj(const char *objFile, obj_mesh_t &mesh) {
  std::ifstream file(objFile);
  if (!file.is_open()) {
    fprintf(stderr, "Error: Could not open file %s\n", objFile);
		return -1;
	}

  mesh.positions.clear();
  mesh.normals.clear();
  mesh.texcoords.clear();
  mesh.vertices.clear();
  mesh.faces.clear();

  std::unordered_map<std::string, uint32_t> vertex_cache;

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") { // Vertex position
      float3_t v;
      iss >> v.x >> v.y >> v.z;
      mesh.positions.push_back(v);
    } else if (type == "vn") { // Vertex normal
      float3_t n;
      iss >> n.x >> n.y >> n.z;
      mesh.normals.push_back(n);
    } else if (type == "vt") { // Texture coordinate (stored as float3 for alignment)
      float2_t tc;
      iss >> tc.x >> tc.y;
      mesh.texcoords.push_back(tc);
    } else if (type == "f") { // Face (triangulated)
      std::vector<std::string> verts;
      std::string vertex;
      while (iss >> vertex)
        verts.push_back(vertex);

      // We build triangles (0, j, j+1) for j=1…verts.size()-2
      for (size_t j = 1; j + 1 < verts.size(); ++j) {
        obj_mesh_t::face_t face;
        std::array<size_t,3> idx = { 0, j, j+1 };
        for (int k = 0; k < 3; ++k) {
          auto parts = split(verts[idx[k]], '/');
          uint32_t p = std::stoi(parts[0]) - 1;
          uint32_t t = (parts.size()>1 && !parts[1].empty()) ? std::stoi(parts[1]) - 1 : 0;
          uint32_t n = (parts.size()>2 && !parts[2].empty()) ? std::stoi(parts[2]) - 1 : 0;
          std::string key = std::to_string(p)+"/"+std::to_string(t)+"/"+std::to_string(n);
          auto it = vertex_cache.find(key);
          if (it != vertex_cache.end()) {
            face.v[k] = it->second;
          } else {
            mesh.vertices.push_back({p,n,t});
            face.v[k] = mesh.vertices.size() - 1;
            vertex_cache[key] = face.v[k];
          }
        }
        mesh.faces.push_back(face);
      }
    }
  }

  return 0; // Success
}

Mesh::Mesh(const char *objFile, const char *texFile, float reflectivity)
  : reflectivity_(reflectivity) {
  obj_mesh_t obj;
	if (load_obj(objFile, obj) != 0) {
		std::abort();
	}

	auto triCount = obj.faces.size();
	tri_.resize(triCount);
	triEx_.resize(triCount);

	for (uint32_t i = 0; i < triCount; i++) {
		const obj_mesh_t::face_t &face = obj.faces[i];
		const obj_mesh_t::vert_t &v0 = obj.vertices[face.v[0]];
		const obj_mesh_t::vert_t &v1 = obj.vertices[face.v[1]];
		const obj_mesh_t::vert_t &v2 = obj.vertices[face.v[2]];
		tri_[i].v0 = obj.positions[v0.p];
		tri_[i].v1 = obj.positions[v1.p];
		tri_[i].v2 = obj.positions[v2.p];

		triEx_[i].N0 = obj.normals[v0.n];
		triEx_[i].N1 = obj.normals[v1.n];
		triEx_[i].N2 = obj.normals[v2.n];

		triEx_[i].uv0 = obj.texcoords[v0.t];
		triEx_[i].uv1 = obj.texcoords[v1.t];
		triEx_[i].uv2 = obj.texcoords[v2.t];
	}

	// load texture
	texture_ = new Surface(texFile);
}

Mesh::~Mesh() {
  delete texture_;
}
