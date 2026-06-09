/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

void inputData(char* fName, int* nx, int* ny, int* nz)
{
  FILE* fid = fopen(fName, "r");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
	
  fread(nx, sizeof(int ),1,fid);
  fread(ny, sizeof(int ),1,fid);
  fread(nz, sizeof(int ),1,fid);
  fclose (fid); 
}

void outputData(char* fName, float *h_A0,int size)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
  tmp32 = size;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  fwrite(h_A0, sizeof(float), tmp32, fid);

  fclose (fid);
}