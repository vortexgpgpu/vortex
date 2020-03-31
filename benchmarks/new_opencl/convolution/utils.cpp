#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

void storeImage(float *imageOut, 
                const char *filename, 
                int rows, 
                int cols, 
                const char* refFilename) {

   FILE *ifp, *ofp;
   unsigned char tmp;
   int offset;
   unsigned char *buffer;
   int i, j;

   int bytes;

   int height, width;

   ifp = fopen(refFilename, "rb");
   if(ifp == NULL) {
      perror(filename);
      exit(-1);
   }

   fseek(ifp, 10, SEEK_SET);
   fread(&offset, 4, 1, ifp);

   fseek(ifp, 18, SEEK_SET);
   fread(&width, 4, 1, ifp);
   fread(&height, 4, 1, ifp);

   fseek(ifp, 0, SEEK_SET);

   buffer = (unsigned char *)malloc(offset);
   if(buffer == NULL) {
      perror("malloc");
      exit(-1);
   }

   fread(buffer, 1, offset, ifp);

   printf("Writing output image to %s\n", filename);
   ofp = fopen(filename, "wb");
   if(ofp == NULL) {
      perror("opening output file");
      exit(-1);
   }
   bytes = fwrite(buffer, 1, offset, ofp);
   if(bytes != offset) {
      printf("error writing header!\n");
      exit(-1);
   }

   // NOTE bmp formats store data in reverse raster order (see comment in
   // readImage function), so we need to flip it upside down here.  
   int mod = width % 4;
   if(mod != 0) {
      mod = 4 - mod;
   }
   //   printf("mod = %d\n", mod);
   for(i = height-1; i >= 0; i--) {
      for(j = 0; j < width; j++) {
         tmp = (unsigned char)imageOut[i*cols+j];
         fwrite(&tmp, sizeof(char), 1, ofp);
      }
      // In bmp format, rows must be a multiple of 4-bytes.  
      // So if we're not at a multiple of 4, add junk padding.
      for(j = 0; j < mod; j++) {
         fwrite(&tmp, sizeof(char), 1, ofp);
      }
   } 

   fclose(ofp);
   fclose(ifp);

   free(buffer);
}

/*
 * Read bmp image and convert to byte array. Also output the width and height
 */
float* readImage(const char *filename, int* widthOut, int* heightOut) {

   uchar* imageData;

   int height, width;
   uchar tmp;
   int offset;
   int i, j;

   printf("Reading input image from %s\n", filename);
   FILE *fp = fopen(filename, "rb");
   if(fp == NULL) {
       perror(filename);
       exit(-1);
   }

   fseek(fp, 10, SEEK_SET);
   fread(&offset, 4, 1, fp);

   fseek(fp, 18, SEEK_SET);
   fread(&width, 4, 1, fp);
   fread(&height, 4, 1, fp);

   printf("width = %d\n", width);
   printf("height = %d\n", height);

   *widthOut = width;
   *heightOut = height;    

   imageData = (uchar*)malloc(width*height);
   if(imageData == NULL) {
       perror("malloc");
       exit(-1);
   }

   fseek(fp, offset, SEEK_SET);
   fflush(NULL);

   int mod = width % 4;
   if(mod != 0) {
       mod = 4 - mod;
   }

   // NOTE bitmaps are stored in upside-down raster order.  So we begin
   // reading from the bottom left pixel, then going from left-to-right, 
   // read from the bottom to the top of the image.  For image analysis, 
   // we want the image to be right-side up, so we'll modify it here.

   // First we read the image in upside-down

   // Read in the actual image
   for(i = 0; i < height; i++) {

      // add actual data to the image
      for(j = 0; j < width; j++) {
         fread(&tmp, sizeof(char), 1, fp);
         imageData[i*width + j] = tmp;
      }
      // For the bmp format, each row has to be a multiple of 4, 
      // so I need to read in the junk data and throw it away
      for(j = 0; j < mod; j++) {
         fread(&tmp, sizeof(char), 1, fp);
      }
   }

   // Then we flip it over
   int flipRow;
   for(i = 0; i < height/2; i++) {
      flipRow = height - (i+1);
      for(j = 0; j < width; j++) {
         tmp = imageData[i*width+j];
         imageData[i*width+j] = imageData[flipRow*width+j];
         imageData[flipRow*width+j] = tmp;
      }
   }

   fclose(fp);

   // Input image on the host
   float* floatImage = NULL;
   floatImage = (float*)malloc(sizeof(float)*width*height);
   if(floatImage == NULL) {
      perror("malloc");
      exit(-1);
   }

   // Convert the BMP image to float (not required)
   for(i = 0; i < height; i++) {
      for(j = 0; j < width; j++) {
         floatImage[i*width+j] = (float)imageData[i*width+j];
      }
   }

   free(imageData);
   return floatImage;
}