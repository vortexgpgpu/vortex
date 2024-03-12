typedef struct __attribute__ ((packed)) vec4f
{
    float x;
    float y;
    float z;
    float w;
};

typedef struct __attribute__ ((packed)) attrib{
    bool enable, normalized;
    unsigned int index;
    int size;
    unsigned int type;
    unsigned int stride;
    const void *pointer;
};

__kernel void vertex_shader (const attrib *VAO,
                      const int _sz_vao,
	                  __global const void *VBO,
                      __global vec4f *P
                      )
{
  int gid = get_global_id(0);
  for(int _i = 0; _i<_sz_vao; ++i) {
    attrib _vao = VAO[_i];
    int _s = 0;
    float* primitive = P[gid];
    for(; _s<_vao.size; ++_s) {
        *(primitive++) = (float) VBO[gid+_s*sizeof(float)];
    }
    while(_s<3) {
        *(primitive++) = 0.f;
        ++_s;
    }
    if(_s <4) *primitive = 1.f;
  }
}
