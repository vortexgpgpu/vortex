#ifndef DWT_CUDA_H
#define	DWT_CUDA_H


namespace dwt_cuda {
  void fdwt53(int * in, int * out, int sizeX, int sizeY, int levels);
//  void rdwt53(int * in, int * out, int sizeX, int sizeY, int levels);
//  void fdwt97(float * in, float * out, int sizeX, int sizeY, int levels);
//  void rdwt97(float * in, float * out, int sizeX, int sizeY, int levels);  
} // namespace dwt_cuda
#endif	// DWT_CUDA_H
