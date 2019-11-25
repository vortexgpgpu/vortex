#pragma once


#ifdef __cplusplus
extern "C" {
#endif

//void vx_vec_sgemm_nn(int n, int m, int k, int* a1, int lda, int* b1, int ldb, int* c1, int ldc);
void vx_vec_sgemm_nn(int n, int m, int k, int* a1, int* b1, int* c1, int ldc, int vsize);
//void vx_vec_sgemm_nn(int n, int* a1, int* b1, int* c1);
#ifdef __cplusplus
}
#endif                   
