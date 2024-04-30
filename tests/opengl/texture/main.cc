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

#define WIDTH 100
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

void createTexturedQuad() {
  static float quad[] = {
    -1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    -1.0, 1.0, 0.0,
    1.0, -1.0, 0.0,
    1.0, 1.0, 0.0,
  };
  static float texture[] = {
    0, 1,
    0, 0,
    1, 0,
    0, 1,
    1, 0,
    1, 1
  };

  GLuint vbo[2];

  glGenBuffers(2, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0); 

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER,sizeof(texture),texture,GL_STATIC_DRAW);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(1); 

}

void createTexture() {
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  PPMImage* image = readPPM("dog.ppm");
  printf("convert\n");
  uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
  for(uint32_t i=0; i<image->x*image->y; ++i) {
    data[i*4+0] = image->data[i*3].red;
    data[i*4+1] = image->data[i*3].green;
    data[i*4+2] = image->data[i*3].blue;
    data[i*4+3] = 0xFFu;
  }

  static int8_t texture_image[256][256][4];
  for(uint32_t r=0; r<=255; ++r)
    for(uint32_t g=0; g<=255; ++g) {
      texture_image[r][g][0] = r;
      texture_image[r][g][1] = g;
      texture_image[r][g][2] = 255;
      texture_image[r][g][3] = 255;
  }

  glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image->x, image->y);
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,image->x,image->y, GL_RGBA, GL_UNSIGNED_BYTE, data);

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

  createTexturedQuad();
  createTexture();
  // Draw
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glFinish();
  glReadnPixels(0,0,WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, WIDTH*HEIGHT*4, result);

  /* TODO
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
  */

  printPPM("image.ppm", WIDTH, HEIGHT, (uint8_t*) result);

  return 0; 
}
