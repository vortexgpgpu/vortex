#include <stdio.h>
#include <vx_print.h>
#include <vx_intrinsics.h>
//#include <iostream>
//#include "vx_mat_mulint32.h"

#define SIZE 2

uint32_t A[SIZE][SIZE] =
{
    {3,5},
    {7,9}
};

uint32_t B[SIZE][SIZE] =
{
    {11,13},
    {15,17}
};



uint32_t Ans[SIZE][SIZE] =
{
    {108,124},
    {212,244}
};

int main() {
	
	int errors = 0;
	vx_printf("KDEBUG Initializing output matrix\n");
    uint32_t C[SIZE][SIZE] =
    {
        {0,0},
        {0,0}
    };
	
	vx_printf("KDEBUG A addr = %x\n", &(A[0][0]));
	//uint32_t a_addr = (uint32_t)A;
	unsigned a_addr = reinterpret_cast<unsigned>(&(A[0][0]));
	unsigned b_addr = reinterpret_cast<unsigned>(&(B[0][0]));
	unsigned c_addr = reinterpret_cast<unsigned>(&(C[0][0]));
	

	vx_printf("KDEBUG a_addr value = %x\n", a_addr);
	/*uint32_t* a_addr = &(A[0][0]);
	uint32_t *b_addr = &(B[0][0]);
	uint32_t *c_addr = &(C[0][0]);*/


	vx_printf("KDEBUG Done Initializing output matrix\n");

	

    	//matmul on vortex
    	// for(int i = 0; i < SIZE; i++){
    	//     for(int j = 0; j < SIZE; j++){
    	//         for(int k = 0; k < SIZE; k++)
    	//         {
    	//             vx_printf("KDEBUG Just before multiply add\n");
    	//             C[i][j] += A[i][k] * B[k][j];
    	// 	    vx_printf("KDEBUG Just after multiply add\n");
    	//         }
    	//     }
    	// }

	// vx_printf("KDEBUG TEST matrix address A = %u, B = %u, C = %u\n", a_addr,  b_addr, c_addr);
    	 ml(0,a_addr);
		 vx_printf("After reinterpret (%x x %d)\n", A[0][0], B[0][0]);
    	 vx_printf("KDEBUG Starting Matmul\n");
		 ml(1,b_addr);
		 
    	 mm();
    	 ms(c_addr);
	// vx_printf("KDEBUG Finished Matmul\n");

	vx_printf("KDEBUG Result of mul C[0][0] = %d\n", C[0][0]);
	vx_printf("KDEBUG Result of mul C[0][0] = %d\n", C[0][1]);
	vx_printf("KDEBUG Result of mul C[0][0] = %d\n", C[1][0]);
	vx_printf("KDEBUG Result of mul C[0][0] = %d\n", C[1][1]);
    	//comparison
	vx_printf("KDEBUG Starting Comparison\n");
    	bool flag = true;
    	for(int i = 0; i < SIZE; i++){
        	for(int j = 0; j < SIZE; j++){
					//vx_printf("KDEBUG Result of mul(%d x%d) = %d\n", A[i][j], B[i][j], C[i][j]);
            		if(C[i][j] != Ans[i][j]){
                		flag = false;
                		break;
            		}
        	}
    	}

	vx_printf("KDEBUG Finished Comparison\n");

	if (flag) {
		vx_printf("Passed!\n");
	} else {
		vx_printf("Failed!");
		errors = 1;
	}

	return errors;
}



