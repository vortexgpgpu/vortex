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

#define WIDTH 10
#define HEIGHT 10

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

GLuint createTriangle() {
  static float triangle[] = {
    0.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0
  };

  GLuint vbo;

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,sizeof(triangle),triangle,GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0); 

  return vbo;
}
GLuint createQuad() {
  static float quad[] = {
    -1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    -1.0, 1.0, 0.0,
    1.0, -1.0, 0.0,
    1.0, 1.0, 0.0,
  };

  GLuint vbo;

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float[3]), 0);
  glEnableVertexAttribArray(0); 

  return vbo;
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

  vbo = createQuad();

  // Draw
  glClear(GL_COLOR_BUFFER_BIT);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glFinish();
  glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);

  printf("Verify result\n");
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

  return errors; 
}
