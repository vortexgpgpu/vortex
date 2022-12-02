//#include <stdlib.h>
//#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <vx_print.h>

#ifdef USE_RISCV_VECTOR
#include "vector_defines.h"
#endif

/*************************************************************************
*GET_TIME
*returns a long int representing the time
*************************************************************************/

//#include <time.h>
//#include <sys/time.h>

/*arv long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
        return (float) (end_time - start_time) / (1000 * 1000);
}*/
/*************************************************************************/
#ifdef USE_RISCV_VECTOR
void axpy_intrinsics(double a, double *dx, double *dy, int n)
{
  int i;

  // long gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
  long gvl = vsetvl_e64m1(n); //PLCT
  
  _MMR_f64 v_a = _MM_SET_f64(a, gvl);
  
  for (i = 0; i < n;) {
    // gvl = __builtin_epi_vsetvl(n - i, __epi_e64, __epi_m1);
    gvl = vsetvl_e64m1(n - i); //PLCT

    _MMR_f64 v_dx = _MM_LOAD_f64(&dx[i], gvl);
    _MMR_f64 v_dy = _MM_LOAD_f64(&dy[i], gvl);
    _MMR_f64 v_res = _MM_MACC_f64(v_dy, v_a, v_dx, gvl);
    _MM_STORE_f64(&dy[i], v_res, gvl);
	
    i += gvl;
  }

  FENCE();
}
#endif

// Ref version
void axpy_ref(double a, double *dx, double *dy, int n)
{
   int i;
   for (i=0; i<n; i++) {
      dy[i] += a*dx[i];
   }
}

void init_vector(double *pv, long n, double value)
{
   for (int i=0; i<n; i++) pv[i]= value;
//   int gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
//   __epi_1xi64 v_value   = __builtin_epi_vbroadcast_1xi64(value, gvl);
//   for (int i=0; i<n; ) {
//    gvl = __builtin_epi_vsetvl(n - i, __epi_e64, __epi_m1);
//      __builtin_epi_vstore_1xf64(&dx[i], v_res, gvl);
//     i += gvl;
//   }
}

void capture_ref_result(double *y, double* y_ref, int n)
{
   int i;
   //printf ("\nReference result: ");
   for (i=0; i<n; i++) {
      y_ref[i]=y[i];
      //printf (" %f", y[i]);
   }
   //printf ("\n\n");
}

void test_result(double *y, double *y_ref, long nrows)
{
   long row;
   int nerrs=0;
   // Compute with the result to keep the compiler for marking the code as dead
   for (row=0; row<nrows; row++)
   {
      double error = y[row] - y_ref[row];
      if(error < 0)
         error = -1 * error;
      if (error > 0.0000001) 
      {
         nerrs++;
         vx_printf("y[%ld]=%.16f != y_ref[%ld]=%.16f  INCORRECT RESULT !!!! \n ", row, y[row], row, y_ref[row]);
         if (nerrs == 100) break;
      }
   }
   if (nerrs == 0)
   {
      vx_printf("Result ok !!!\n");
   }
}

int main(/*arv int argc, char *argv[]*/)
{
    //long long start,end;
    //arv start = get_time();

    double a = 1.0;
    const long n = 2;

    /*arv if (argc == 2)
	    n = 1024*atol(argv[1]); // input argument: vector size in Ks
    else
        n = (30*1024);*/


    /* Allocate the source and result vectors */
    /*arv double *dx     = (double*)malloc(n*sizeof(double));
    double *dy     = (double*)malloc(n*sizeof(double));
    double *dy_ref = (double*)malloc(n*sizeof(double));*/
    double dx[n], dy[n], dy_ref[n];

    init_vector(dx, n, 1.0);
    init_vector(dy, n, 2.0);
    
    //arv end = get_time();
    //arv printf("init_vector time: %f\n", elapsed_time(start, end));

    vx_printf("doing reference axpy\n");
    //arv start = get_time();
    axpy_ref(a, dx, dy, n);

    //arv end = get_time();
    //arv printf("axpy_reference time: %f\n", elapsed_time(start, end));

    capture_ref_result(dy, dy_ref, n);
    
    init_vector(dx, n, 1.0);
    init_vector(dy, n, 2.0);

    vx_printf ("doing vector axpy\n");
    //arv start = get_time();
    #ifdef USE_RISCV_VECTOR
    axpy_intrinsics(a, dx, dy, n);
    #endif
    //arv end = get_time();
    //arv printf("axpy_intrinsics time: %f\n", elapsed_time(start, end));
    
    vx_printf ("done\n");
    test_result(dy, dy_ref, n);

   //arv free(dx); free(dy); free(dy_ref);
   return 0;
}
