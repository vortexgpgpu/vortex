#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec_sfilter.h"

#define NUM_DATA 66

int main()
{
    vx_tmc(1);

    int n = NUM_DATA*NUM_DATA;
    int ldc = NUM_DATA;

    /*int m0 = 1;
    int m1 = 1;
    int m2 = 1;
    int m3 = 1;
    int m4 = 1;
    int m5 = 1;
    int m6 = 1;
    int m7 = 1;
    int m8 = 1;*/
    int m = 1;

    int *a = (int*)malloc(sizeof(int) * n); 
    int *b = (int*)malloc(sizeof(int) * n); 
    int *c = (int*)malloc(sizeof(int) * n); 


    for (int i = 0; i < n; ++i) { 
        a[i] = i;
        b[i] = 0; 
        c[i] = 0;
    }

    int N = 4;
    int startCycles = vx_num_cycles();
    int startInst = vx_num_instrs();
    for(int y = 1; y < (NUM_DATA-1); y++){
        for(int x = 1; x < (NUM_DATA-1); x = x+N) {
            vx_vec_sfilter(a, b, ldc, m, x, y, N);
        }
    }
    int endCycles = vx_num_cycles();
    int endInst = vx_num_instrs();

    int totalInst = (endInst - startInst);
    int totalCycles = (endCycles - startCycles);

    printf("\nCycles = %d, Instructions = %d", totalCycles, totalInst);
    

    for(int y = 1; y < (NUM_DATA-1); ++y) 
    {
    	for(int x = 1; x < (NUM_DATA-1); ++x){
	      int i0 = a[(x-1)+(y-1)*ldc]*m;
          //printf("a[%d] = %d",((x-1)+(y-1)*ldc), a[(x-1)+(y-1)*ldc] );
		  int i1 = a[(x)  +(y-1)*ldc]*m;
		  int i2 = a[(x+1)+(y-1)*ldc]*m;
		  int i3 = a[(x-1)+(y)  *ldc]*m;
		  int i4 = a[(x)  + y  * ldc]*m;
		  int i5 = a[(x+1)+(y)  *ldc]*m;
		  int i6 = a[(x-1)+(y+1)*ldc]*m;
		  int i7 = a[(x)  +(y+1)*ldc]*m;
		  int i8 = a[(x+1)+(y+1)*ldc]*m;

	  	  c[x+y*ldc] = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
          //printf("\nc[%d] = %d",(x+y*ldc), c[x+y*ldc] );
          //printf("\nb[%d] = %d",(x+y*ldc), b[x+y*ldc] );
          //printf("%d, %d, %d, %d, %d, %d, %d, %d, %d", i0, i1, i2, i3, i4, i5, i6, i7, i8);
	      if(c[x+y*ldc] != b[x+y*ldc] ) 
	      {
	        printf("\n<saxpy> FAILED at <index: %d>! \n", x);
	        return 1;   
	      }
  		}
    }
    
    printf("\nPASSED.......................... <sfilter> \n");


    free(a); free(b); free(c);

    vx_tmc(0);

    return 0;
}
