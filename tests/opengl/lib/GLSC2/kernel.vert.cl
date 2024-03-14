typedef struct __attribute__ ((packed)) vert_attrib
{
  bool enable, normalized;
  GLuint index;
  GLint size;
  GLenum type;
  GLsizei stride;
  const void *pointer;
};

__kernel void vertex_shader (
                      unsigned int _sz,
                      vert_attrib *_VAO,
                      __global const float* VBO,
                      __global void* P
                      )
{
  int gid = get_global_id(0);

  vert_attrib *END = _VAO+_sz;
  while (_VAO != END) {
    if (_VAO)
  }
  for(unsigned int vao = 0; vao < _sz; ++vao) {

  }
  float* read_pos = &VBO[first*size + gid*size + (unsigned int)(offset/sizeof(float))];
  float* write_pos = &P[4*gid];

  for (int i=0; i<size; i++){
    write_pos[i]=*read_pos++;
  }

  //llenamos vector output
  while (size < 4){
    write_pos[size]=0;
    if(size==3)
      write_pos[size] = 1;
    size++;
  }
}
