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

#define WIDTH 75
#define HEIGHT 50

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
  glBufferData(GL_ARRAY_BUFFER,sizeof(color),color,GL_STATIC_DRAW);
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
  GLfloat view[16];
  GLfloat model[16];

  perspectiveMatrix(perspective, M_PI / 2.f, (float) WIDTH / (float) HEIGHT, 0.0f, 1.0f);
  // TODO: glGetUniformLocation(program, "perspective");
  glUniformMatrix4fv(0, 4, GL_FALSE, perspective);

  float eye[3] = {0.f, 0.f, -1.f}; 
  float center[3] = {0.f, 0.f, 0.f}; 
  float up[3] = {0.f, 1.f, 0.f}; 
  lookAtRH(view, eye, center, up);
  // TODO: glGetUniformLocation(program, "view");
  glUniformMatrix4fv(1, 4, GL_FALSE, view);
  glEnable(GL_DEPTH_TEST);
  // Draw
  uint rotation = 0;
  while (rotation < 5) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rotateMatrix(model, M_PI/4*(rotation++), 1,1,1);
    glUniformMatrix4fv(2, 4, GL_FALSE, model);

    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
    glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);
    printPPM("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);
  }

  return 0; 
}

void perspectiveMatrix(float* matrix, float angle, float ratio, float near, float far) {
    float f = 1.0f / tan(angle / 2.0f);
    matrix[0] = f / ratio;
    matrix[1] = 0.0f;
    matrix[2] = 0.0f;
    matrix[3] = 0.0f;
    matrix[4] = 0.0f;
    matrix[5] = f;
    matrix[6] = 0.0f;
    matrix[7] = 0.0f;
    matrix[8] = 0.0f;
    matrix[9] = 0.0f;
    matrix[10] = (far + near) / (near - far);
    matrix[11] = (2.0f * far * near) / (near - far);
    matrix[12] = 0.0f;
    matrix[13] = 0.0f;
    matrix[14] = -1.0f;
    matrix[15] = 0.0f;
}

void rotateMatrix(float *matrix, float angle, float x, float y, float z) {
    float c = cos(angle);
    float s = sin(angle);
    float one_c = 1.0f - c;

    float mag = sqrt(x * x + y * y + z * z);
    if (mag > 0.0f) {
        x /= mag;
        y /= mag;
        z /= mag;
    }
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float zx = z * x;
    float xs = x * s;
    float ys = y * s;
    float zs = z * s;

    matrix[0] = (one_c * xx) + c;
    matrix[1] = (one_c * xy) - zs;
    matrix[2] = (one_c * zx) + ys;
    matrix[3] = 0.0f;

    matrix[4] = (one_c * xy) + zs;
    matrix[5] = (one_c * yy) + c;
    matrix[6] = (one_c * yz) - xs;
    matrix[7] = 0.0f;

    matrix[8] = (one_c * zx) - ys;
    matrix[9] = (one_c * yz) + xs;
    matrix[10] = (one_c * zz) + c;
    matrix[11] = 0.0f;

    matrix[12] = 0.0f;
    matrix[13] = 0.0f;
    matrix[14] = 0.0f;
    matrix[15] = 1.0f;
}

#define DOT(a, b) a[0]*b[0]+a[1]*b[1]+a[2]*b[2] 

void lookAtRH(float *matrix, const float eye[3], const float center[3], const float up[3]) {
  
  float f[3];
  f[0] = center[0] - eye[0];
  f[1] = center[1] - eye[1];
  f[2] = center[2] - eye[2];
  float mag_f = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
  f[0] /= mag_f; f[1] /= mag_f; f[2] /= mag_f;

  float s[3];
  s[0] = f[1]*up[2] - f[2]*up[1];
  s[1] = f[2]*up[0] - f[0]*up[2];
  s[2] = f[0]*up[1] - f[1]*up[0];
  float mag_s = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
  s[0] /= mag_s; s[1] /= mag_s; s[2] /= mag_s;

  float u[3];
  u[0] = s[1]*f[2] - s[2]*f[1];
  u[1] = s[2]*f[0] - s[0]*f[2];
  u[2] = s[0]*f[1] - s[1]*f[0];

	matrix[0] = s[0];
	matrix[1] = u[0];
	matrix[2] =-f[0];
	matrix[3] = 0.f;
	matrix[4] = s[1];
	matrix[5] = u[1];
	matrix[6] =-f[1];
	matrix[8] = s[2];
	matrix[9] = u[2];
	matrix[10] =-f[2];
	matrix[11] = 0.f;
	matrix[12] =-DOT(s, eye);
	matrix[13] =-DOT(u, eye);
	matrix[14] = DOT(f, eye);
	matrix[15] = 1.f;
}