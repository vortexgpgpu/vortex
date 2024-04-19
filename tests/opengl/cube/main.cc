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

#define WIDTH 20
#define HEIGHT 20

void perspectiveMatrix(float* mat, float angle, float ratio, float near, float far);
void rotateMatrix(float* mat, float angle, float x, float y, float z);

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

void createCube() {
  static float cube[] = {
    // FRONT
    -0.5, 0.5, -0.5,
    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    -0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    0.5, -0.5, -0.5,
    // BACK
    -0.5, 0.5, 0.5,
    -0.5, -0.5, 0.5,
    0.5, -0.5, 0.5,
    -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, -0.5, 0.5,
    // TOP
    -0.5, 0.5, 0.5,
    -0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, -0.5,
    // BOTTOM
    -0.5, -0.5, 0.5,
    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5,
    0.5, -0.5, 0.5,
    0.5, -0.5, -0.5,
    // LEFT
    -0.5, 0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5,
    -0.5, 0.5, -0.5,
    -0.5, 0.5, 0.5,
    -0.5, -0.5, 0.5,
    // RIGHT
    0.5, 0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5, 0.5,
    0.5, 0.5, -0.5,
    0.5, 0.5, 0.5,
    0.5, -0.5, 0.5,
  };

  static float color[] = {
    // FRONT
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    // BACK
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    // TOP
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    // BOTTOM
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    // LEFT
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    // RIGHT
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  };

  GLuint vbo[2];

  glGenBuffers(2, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(cube),cube,GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float[3]), 0);
  glEnableVertexAttribArray(0); 

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(cube),cube,GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float[3]), 0);
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

  createCube();

  GLfloat perspective[16];
  GLfloat model[16];

  perspectiveMatrix(perspective, M_PI / 2, (float) WIDTH / (float) HEIGHT, 0.0f, 1.0f);
  glUniformMatrix4fv(0, 4, GL_FALSE, perspective);
  // Draw
  uint rotation = 0;
  while (true) {
    glClear(GL_COLOR_BUFFER_BIT);

    rotateMatrix(model, M_PI/6*(rotation++), 0,1,0);
    glUniformMatrix4fv(1, 4, GL_FALSE, model);

    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
    glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);
    printPPM("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);
  }

  return 0; 
}

void perspectiveMatrix(float* mat, float angle, float ratio, float near, float far) {
  for(int i=0; i<16; ++i) mat[i]=0.f;
  
  float tan_half_angle = tan(angle/2);
  mat[0] = 1.0f / (ratio * tan_half_angle);
  mat[5] = 1.0f / tan_half_angle;
  mat[10] = -(far + near) / (far - near);
  mat[11] = -(2 * far * near) / (far - near);
  mat[14] = -1.0f;
}

void rotateMatrix(float *mat, float angle, float x, float y, float z) {
  for(int i=0; i<16; ++i) mat[i]=0.f;

  float sx, sy, sz, cx, cy, cz;
  sx = sin(angle*x);
  sy = sin(angle*y);
  sz = sin(angle*z);
  cx = cos(angle*x);
  cy = cos(angle*y);
  cz = cos(angle*z);

  mat[0] = cy*cx;
  mat[1] = sz*sy*cx - cz*sy;
  mat[2] = cz*sy*cx + sz*sx;
  mat[4] = cy*sx;
  mat[5] = sz*sy*sx + cy*cx;
  mat[6] = cz*sy*sx - sz*cx;
  mat[8] = -sy;
  mat[9] = sz*cy;
  mat[10] = cz*cy;
  mat[15] = 1.f;
}