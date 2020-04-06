#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#include "utils.h"

// This function takes a positive integer and rounds it up to
// the nearest multiple of another provided integer
unsigned int roundUp(unsigned int value, unsigned int multiple) {

  // Determine how far past the nearest multiple the value is
  unsigned int remainder = value % multiple;

  // Add the difference to make the value a multiple
  if(remainder != 0) {
          value += (multiple-remainder);
  }

  return value;
}

// This function reads in a text file and stores it as a char pointer
char* readSource(char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   printf("Program file is: %s\n", kernelPath);

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);

   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   }

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
}

int main() {

   int i, j, k, l;

   // Rows and columns in the input image
   int imageHeight;
   int imageWidth;

   const char* inputFile = "input.bmp";
   const char* outputFile = "output.bmp";

   // Homegrown function to read a BMP from file
   float* inputImage = readImage(inputFile, &imageWidth,
      &imageHeight);

   // Size of the input and output images on the host
   int dataSize = imageHeight*imageWidth*sizeof(float);

   // Output image on the host
   float* outputImage = NULL;
   outputImage = (float*)malloc(dataSize);
   float* refImage = NULL;
   refImage = (float*)malloc(dataSize);

   // 45 degree motion blur
   float filter[49] =
      {0,      0,      0,      0,      0,      0,      0,
       0,      0,      0,      0,      0,      0,      0,
       0,      0,     -1,      0,      1,      0,      0,
       0,      0,     -2,      0,      2,      0,      0,
       0,      0,     -1,      0,      1,      0,      0,
       0,      0,      0,      0,      0,      0,      0,
       0,      0,      0,      0,      0,      0,      0};

   // The convolution filter is 7x7
   int filterWidth = 7;  
   int filterSize  = filterWidth*filterWidth;  // Assume a square kernel

   // Set up the OpenCL environment
   cl_int status;

   // Discovery platform
   cl_platform_id platform;
   status = clGetPlatformIDs(1, &platform, NULL);
   chk(status, "clGetPlatformIDs");

   // Discover device
   cl_device_id device;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
   chk(status, "clGetDeviceIDs");

   // Create context
   cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
       (cl_context_properties)(platform), 0};
   cl_context context;
   context = clCreateContext(props, 1, &device, NULL, NULL, &status);
   chk(status, "clCreateContext");

   // Create command queue
   cl_command_queue queue;
   queue = clCreateCommandQueue(context, device, 0, &status);
   chk(status, "clCreateCommandQueue");

   // The image format describes how the data will be stored in memory
   cl_image_format format;
   format.image_channel_order     = CL_R;     // single channel
   format.image_channel_data_type = CL_FLOAT; // float data type

   // Create space for the source image on the device
   cl_mem d_inputImage = clCreateImage2D(context, 0, &format, imageWidth, 
      imageHeight, 0, NULL, &status);
   chk(status, "clCreateImage2D");

   // Create space for the output image on the device
   cl_mem d_outputImage = clCreateImage2D(context, 0, &format, imageWidth, 
      imageHeight, 0, NULL, &status);
   chk(status, "clCreateImage2D");

   // Create space for the 7x7 filter on the device
   cl_mem d_filter = clCreateBuffer(context, 0, filterSize*sizeof(float), 
      NULL, &status);
   chk(status, "clCreateBuffer");

   // Copy the source image to the device
   size_t origin[3] = {0, 0, 0};  // Offset within the image to copy from
   size_t region[3] = {imageWidth, imageHeight, 1}; // Elements to per dimension
   status = clEnqueueWriteImage(queue, d_inputImage, CL_FALSE, origin, region, 
      0, 0, inputImage, 0, NULL, NULL);
   chk(status, "clEnqueueWriteImage");
    
   // Copy the 7x7 filter to the device
   status = clEnqueueWriteBuffer(queue, d_filter, CL_FALSE, 0, 
      filterSize*sizeof(float), filter, 0, NULL, NULL);
   chk(status, "clEnqueueWriteBuffer");

   // Create the image sampler
   cl_sampler sampler = clCreateSampler(context, CL_FALSE, 
      CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
   chk(status, "clCreateSampler");

   const char* source = readSource("kernel.cl");

   // Create a program object with source and build it
   cl_program program;
   program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
   chk(status, "clCreateProgramWithSource");
   status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   chk(status, "clBuildProgram");
 
   // Create the kernel object
   cl_kernel kernel;
   kernel = clCreateKernel(program, "convolution", &status);
   chk(status, "clCreateKernel");

   // Set the kernel arguments
   status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputImage);
   status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_outputImage);
   status |= clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
   status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
   status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_filter);
   status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
   status |= clSetKernelArg(kernel, 6, sizeof(cl_sampler), &sampler);
   chk(status, "clSetKernelArg");

   // Set the work item dimensions
   size_t globalSize[2] = {imageWidth, imageHeight};
   status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0,
      NULL, NULL);
   chk(status, "clEnqueueNDRange");

   // Read the image back to the host
   status = clEnqueueReadImage(queue, d_outputImage, CL_TRUE, origin, 
      region, 0, 0, outputImage, 0, NULL, NULL); 
   chk(status, "clEnqueueReadImage");

   // Write the output image to file
   storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);

   // Compute the reference image
   for(i = 0; i < imageHeight; i++) {
      for(j = 0; j < imageWidth; j++) {
         refImage[i*imageWidth+j] = 0;
      }
   }

   // Iterate over the rows of the source image
   int halfFilterWidth = filterWidth/2;
   float sum;
   for(i = 0; i < imageHeight; i++) {
      // Iterate over the columns of the source image
      for(j = 0; j < imageWidth; j++) {
         sum = 0; // Reset sum for new source pixel
         // Apply the filter to the neighborhood
         for(k = - halfFilterWidth; k <= halfFilterWidth; k++) {
            for(l = - halfFilterWidth; l <= halfFilterWidth; l++) {
               if(i+k >= 0 && i+k < imageHeight && 
                  j+l >= 0 && j+l < imageWidth) {
                  sum += inputImage[(i+k)*imageWidth + j+l] * 
                         filter[(k+halfFilterWidth)*filterWidth + 
                            l+halfFilterWidth];
               }
            }
         }
         refImage[i*imageWidth+j] = sum;
      }
   }

   int failed = 0;
   for(i = 0; i < imageHeight; i++) {
      for(j = 0; j < imageWidth; j++) {
         if(abs(outputImage[i*imageWidth+j]-refImage[i*imageWidth+j]) > 0.01) {
            printf("Results are INCORRECT\n");
            printf("Pixel mismatch at <%d,%d> (%f vs. %f)\n", i, j,
               outputImage[i*imageWidth+j], refImage[i*imageWidth+j]);
            failed = 1;
         }
         if(failed) break;
      }
      if(failed) break;
   }
   if(!failed) {
      printf("Results are correct\n");
   }
             
   return 0;
}