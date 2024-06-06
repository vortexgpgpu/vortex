#include <GLSC2/glsc2.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <GLSC2/glsc2.h>
#include <unistd.h> 
#include <string.h>

/** Kernel utils */

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (NULL == filename || NULL == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}


/** Context utils **\
 * 
*/
typedef struct {
    GLuint colorbuffer, depthbuffer, stencilbuffer;
} RENDERBUFFERS;

typedef struct {
    GLuint framebuffers[2]; // buffers to be swapped
    RENDERBUFFERS renderbuffers[2]; // renderbuffer associeted
    GLsizei activated; // framebuffer active
} CONTEXT;

void createContext(CONTEXT* display, GLuint width, GLuint height) {

    // Gen framebuffers & renderbuffers
    glGenFramebuffers(2, display->framebuffers);
    glGenRenderbuffers(1, &display->renderbuffers[0].colorbuffer);
    glGenRenderbuffers(1, &display->renderbuffers[0].depthbuffer);
    glGenRenderbuffers(1, &display->renderbuffers[0].stencilbuffer);
    glGenRenderbuffers(1, &display->renderbuffers[1].colorbuffer);
    glGenRenderbuffers(1, &display->renderbuffers[1].depthbuffer);
    glGenRenderbuffers(1, &display->renderbuffers[1].stencilbuffer);

    // Framebuffer0
    glBindFramebuffer(GL_FRAMEBUFFER, display->framebuffers[0]);
    // Build the renderbuffers
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[0].colorbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[0].depthbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[0].stencilbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, width, height);
    // Associate buffers
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, display->renderbuffers[0].colorbuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, display->renderbuffers[0].depthbuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, display->renderbuffers[0].stencilbuffer);

    // Framebuffer1
    glBindFramebuffer(GL_FRAMEBUFFER, display->framebuffers[1]);
    // Build the renderbuffers
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[1].colorbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[1].depthbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, display->renderbuffers[1].stencilbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, width, height);
    // Associate buffers
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, display->renderbuffers[1].colorbuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, display->renderbuffers[1].depthbuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, display->renderbuffers[1].stencilbuffer);
}

void swapBuffers(CONTEXT* display) {
    GLsizei to_activate = 1;
    if (display->activated == 1) to_activate = 0;

    glBindFramebuffer(GL_FRAMEBUFFER ,display->framebuffers[to_activate]);
    display->activated = to_activate;
}

/* Image utils */
void printPPM(const char* filename, size_t width, size_t height, const uint8_t *data) {
  FILE *f = fopen(filename, "wb");
  fprintf(f, "P6\n%i %i 255\n", width, height);
  for (int y=0; y<height; y++) {
      for (int x=0; x<width; x++) {
          fputc(data[0], f); 
          fputc(data[1], f); // 0 .. 255
          fputc(data[2], f);  // 0 .. 255
          data += 4;
      }
  }
  fclose(f);
}

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}



