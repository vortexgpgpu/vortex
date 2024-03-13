__kernel void vertex_shader (__global unsigned int first,
                      __global const float* pos,
                      __global const void* pos_offset,
                      __global const unsigned_int pos_size,
                      __global float* P
                      )
{
  int gid = get_global_id(0);

  float* read_pos = &pos[first*size + gid*size + (unsigned int)(offset/sizeof(float))];
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
