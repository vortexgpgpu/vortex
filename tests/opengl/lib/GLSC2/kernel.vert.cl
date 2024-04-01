struct VAO{
    bool enable, normalized;
    unsigned int index;
    int size;
    unsigned int type;
    unsigned int stride;
    const void *pointer;
};

struct vec3f{
  float x;
  float y;
  float z;
}

struct vec4f{
  float x;
  float y;
  float z;
  float w;
}

void vertexatrib3f(int index, int gid, float* x, float* y, float* z, VAO* vao, float* vbo){
  VAO i_vao=vao[index];
  float* init = vbo[first*i_vao->size + gid*i_vao->size + (unsigned int) i_vao->pointer];
  *x = *init++;
  *y = *init++;
  *z = *init;
}

__kernel void vertex_shader (__global unsigned int first,
                      __global const void* vao,
                      __global void* primitives,
                      __global const void* vbo
                      )
{
  int gid = get_global_id(0);
  vec3f position;
  vertexatrib3f(0, gid, &position.x, &position.y, &position.z, (VAO*) vao, (float*) vbo);

  vec4* gl_position = ((float*) primitives)[4*gid];

  gl_position.x = position.x;
  gl_position.y = position.y;
  gl_position.z = position.z;
  gl_position.w = 1;
}
