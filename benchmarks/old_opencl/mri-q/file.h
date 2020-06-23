/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void inputData(char* fName, int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI);

void outputData(char* fName, float* outR, float* outI, int numX);

#ifdef __cplusplus
}
#endif
