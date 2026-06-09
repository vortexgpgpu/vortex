/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

//#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

void inputData(char* fName, int* len, int* depth, int* dim,int *nzcnt_len,int *pad,
               float** h_data, int** h_indices, int** h_ptr,
               int** h_perm, int** h_nzcnt)
{
  FILE* fid = fopen(fName, "rb");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }

  fscanf(fid, "%d %d %d %d %d\n",len,depth,nzcnt_len,dim,pad);
  int _len=len[0];
  int _depth=depth[0];
  int _dim=dim[0];
  int _pad=pad[0];
  int _nzcnt_len=nzcnt_len[0];
  
  *h_data = (float *) malloc(_len * sizeof (float));
  fread (*h_data, sizeof (float), _len, fid);
  
  *h_indices = (int *) malloc(_len * sizeof (int));
  fread (*h_indices, sizeof (int), _len, fid);
  
  *h_ptr = (int *) malloc(_depth * sizeof (int));
  fread (*h_ptr, sizeof (int), _depth, fid);
  
  *h_perm = (int *) malloc(_dim * sizeof (int));
  fread (*h_perm, sizeof (int), _dim, fid);
  
  *h_nzcnt = (int *) malloc(_nzcnt_len * sizeof (int));
  fread (*h_nzcnt, sizeof (int), _nzcnt_len, fid);

  fclose (fid); 
}

void input_vec(char *fName,float *h_vec,int dim)
{
  FILE* fid = fopen(fName, "rb");
  fread (h_vec, sizeof (float), dim, fid);
  fclose(fid);
  
}

void outputData(char* fName, float *h_Ax_vector,int dim)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
  tmp32 = dim;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  fwrite(h_Ax_vector, sizeof(float), dim, fid);

  fclose (fid);
}
