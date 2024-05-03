#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <GLSC2/glsc2.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>

#include "../debug.cc"
#include "../common.c"

#define WIDTH 150
#define HEIGHT 100

GLuint createProgram(const char* filename) {
  GLuint program;
  size_t kernel_size;
  uint8_t *kernel_bin;

  program = glCreateProgram();
  
  if (0 != read_kernel_file(filename, &kernel_bin, &kernel_size))
    return -1;
  
  glProgramBinary (program, 0, (void*) kernel_bin, kernel_size);
  
  return program;
}

void createTriangle() {
  static float triangle[] = {
    0.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0
  };
  static float color[] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  };

  GLuint vbo[2];

  glGenBuffers(2, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(triangle),triangle,GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(0); 

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(color),color,GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glEnableVertexAttribArray(1); 
}

int main() {
  // Set up vertex buffer object
  CONTEXT context;
  unsigned char result[WIDTH][HEIGHT][4]; // RGBA8 32 bits x fragment
  GLuint vbo, program;

  // TODO: linker issue
  cl_platform_id platform_id;
  clGetPlatformIDs(1, &platform_id, NULL);

  // Set up context
  createContext(&context, WIDTH, HEIGHT);
  glViewport(0, 0, WIDTH, HEIGHT); 

  // Create program & vbo
  program = createProgram("kernel.pocl");
  glUseProgram(program);

  createTriangle();

  // Draw
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glFinish();
  glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);

  printf("Verify result\n");

  /* TODO: this is for quad not for triangle
  int errors = 0;
  unsigned int *rgba8 = (unsigned int*) result;
  for (int i = 0; i < WIDTH*HEIGHT; ++i) {
    unsigned int ref = 0xFFFFFFFF;
    if (rgba8[i] != ref) {
      if (errors < 100) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, rgba8[i]);
      ++errors;
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);    
  }
  */

  printPPM("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);

  return 0; 
}
