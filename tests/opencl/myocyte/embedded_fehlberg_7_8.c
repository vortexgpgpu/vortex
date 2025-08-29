// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DESCRIPTION
//========================================================================================================================================================================================================200

////////////////////////////////////////////////////////////////////////////////
// File: embedded_fehlberg_7_8.c                                              //
// Routines:                                                                  //
//    Embedded_Fehlberg_7_8                                                   //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Description:                                                              //
//     The Runge-Kutta-Fehlberg method is an adaptive procedure for approxi-  //
//     mating the solution of the differential equation y'(x) = f(x,y) with   //
//     initial condition y(x0) = c.  This implementation evaluates f(x,y)     //
//     thirteen times per step using embedded seventh order and eight order   //
//     Runge-Kutta estimates to estimate the not only the solution but also   //
//     the error.                                                             //
//     The next step size is then calculated using the preassigned tolerance  //
//     and error estimate.                                                    //
//     For step i+1,                                                          //
//        y[i+1] = y[i] +  h * (41/840 * k1 + 34/105 * finavalu_temp[5] + 9/35 * finavalu_temp[6]         //
//                        + 9/35 * finavalu_temp[7] + 9/280 * finavalu_temp[8] + 9/280 finavalu_temp[9] + 41/840 finavalu_temp[10] ) //
//     where                                                                  //
//     k1 = f( x[i],y[i] ),                                                   //
//     finavalu_temp[1] = f( x[i]+2h/27, y[i] + 2h*k1/27),                                  //
//     finavalu_temp[2] = f( x[i]+h/9, y[i]+h/36*( k1 + 3 finavalu_temp[1]) ),                            //
//     finavalu_temp[3] = f( x[i]+h/6, y[i]+h/24*( k1 + 3 finavalu_temp[2]) ),                            //
//     finavalu_temp[4] = f( x[i]+5h/12, y[i]+h/48*(20 k1 - 75 finavalu_temp[2] + 75 finavalu_temp[3])),                //
//     finavalu_temp[5] = f( x[i]+h/2, y[i]+h/20*( k1 + 5 finavalu_temp[3] + 4 finavalu_temp[4] ) ),                    //
//     finavalu_temp[6] = f( x[i]+5h/6, y[i]+h/108*( -25 k1 + 125 finavalu_temp[3] - 260 finavalu_temp[4] + 250 finavalu_temp[5] ) ), //
//     finavalu_temp[7] = f( x[i]+h/6, y[i]+h*( 31/300 k1 + 61/225 finavalu_temp[4] - 2/9 finavalu_temp[5]              //
//                                                            + 13/900 finavalu_temp[6]) )  //
//     finavalu_temp[8] = f( x[i]+2h/3, y[i]+h*( 2 k1 - 53/6 finavalu_temp[3] + 704/45 finavalu_temp[4] - 107/9 finavalu_temp[5]      //
//                                                      + 67/90 finavalu_temp[6] + 3 finavalu_temp[7]) ), //
//     finavalu_temp[9] = f( x[i]+h/3, y[i]+h*( -91/108 k1 + 23/108 finavalu_temp[3] - 976/135 finavalu_temp[4]        //
//                             + 311/54 finavalu_temp[5] - 19/60 finavalu_temp[6] + 17/6 finavalu_temp[7] - 1/12 finavalu_temp[8]) ), //
//     finavalu_temp[10] = f( x[i]+h, y[i]+h*( 2383/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]     //
//          - 301/82 finavalu_temp[5] + 2133/4100 finavalu_temp[6] + 45/82 finavalu_temp[7] + 45/164 finavalu_temp[8] + 18/41 finavalu_temp[9]) )  //
//     finavalu_temp[11] = f( x[i], y[i]+h*( 3/205 k1 - 6/41 finavalu_temp[5] - 3/205 finavalu_temp[6] - 3/41 finavalu_temp[7]        //
//                                                   + 3/41 finavalu_temp[8] + 6/41 finavalu_temp[9]) )  //
//     finavalu_temp[12] = f( x[i]+h, y[i]+h*( -1777/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]    //
//                      - 289/82 finavalu_temp[5] + 2193/4100 finavalu_temp[6] + 51/82 finavalu_temp[7] + 33/164 finavalu_temp[8] +   //
//                                                        12/41 finavalu_temp[9] + finavalu_temp[11]) )  //
//     x[i+1] = x[i] + h.                                                     //
//                                                                            //
//     The error is estimated to be                                           //
//        err = -41/840 * h * ( k1 + finavalu_temp[10] - finavalu_temp[11] - finavalu_temp[12])                         //
//     The step size h is then scaled by the scale factor                     //
//         scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     //
//     The scale factor is further constrained 0.125 < scale < 4.0.           //
//     The new step size is h := scale * h.                                   //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  static fp Runge_Kutta(fp (*f)(fp,fp), fp *y, fp x0, fp h)                 //
//                                                                            //
//  Description:                                                              //
//     This routine uses Fehlberg's embedded 7th and 8th order methods to     //
//     approximate the solution of the differential equation y'=f(x,y) with   //
//     the initial condition y = y[0] at x = x0.  The value at x + h is       //
//     returned in y[1].  The function returns err / h ( the absolute error   //
//     per step size ).                                                       //
//                                                                            //
//  Arguments:                                                                //
//     fp *f  Pointer to the function which returns the slope at (x,y) of     //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y[0]).                   //
//     fp y[] On input y[0] is the initial value of y at x, on output         //
//                y[1] is the solution at x + h.                              //
//     fp x   Initial value of x.                                             //
//     fp h   Step size                                                       //
//                                                                            //
//  Return Values:                                                            //
//     This routine returns the err / h.  The solution of y(x) at x + h is    //
//     returned in y[1].                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "master.h"															// (in directory)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <math.h>															// (in path provided to compiler)	needed by pow, fabs
#include <stdlib.h>															// (in path provided to compiler)	needed by malloc, free

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	PARTICULAR SOLVER FUNCTION
//========================================================================================================================================================================================================200

void 
embedded_fehlberg_7_8(	fp timeinst,
						fp h,
						fp *initvalu,
						fp *finavalu,
						fp *error,
						fp *parameter,
						fp *com,

						cl_mem d_initvalu,
						cl_mem d_finavalu,
						cl_mem d_params,
						cl_mem d_com,

						cl_command_queue command_queue,
						cl_kernel kernel,

						long long *timecopyin,
						long long *timecopykernel,
						long long *timecopyout) 
{

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	static const fp c_1_11 = 41.0 / 840.0;
	static const fp c6 = 34.0 / 105.0;
	static const fp c_7_8= 9.0 / 35.0;
	static const fp c_9_10 = 9.0 / 280.0;

	static const fp a2 = 2.0 / 27.0;
	static const fp a3 = 1.0 / 9.0;
	static const fp a4 = 1.0 / 6.0;
	static const fp a5 = 5.0 / 12.0;
	static const fp a6 = 1.0 / 2.0;
	static const fp a7 = 5.0 / 6.0;
	static const fp a8 = 1.0 / 6.0;
	static const fp a9 = 2.0 / 3.0;
	static const fp a10 = 1.0 / 3.0;

	static const fp b31 = 1.0 / 36.0;
	static const fp b32 = 3.0 / 36.0;
	static const fp b41 = 1.0 / 24.0;
	static const fp b43 = 3.0 / 24.0;
	static const fp b51 = 20.0 / 48.0;
	static const fp b53 = -75.0 / 48.0;
	static const fp b54 = 75.0 / 48.0;
	static const fp b61 = 1.0 / 20.0;
	static const fp b64 = 5.0 / 20.0;
	static const fp b65 = 4.0 / 20.0;
	static const fp b71 = -25.0 / 108.0;
	static const fp b74 =  125.0 / 108.0;
	static const fp b75 = -260.0 / 108.0;
	static const fp b76 =  250.0 / 108.0;
	static const fp b81 = 31.0/300.0;
	static const fp b85 = 61.0/225.0;
	static const fp b86 = -2.0/9.0;
	static const fp b87 = 13.0/900.0;
	static const fp b91 = 2.0;
	static const fp b94 = -53.0/6.0;
	static const fp b95 = 704.0 / 45.0;
	static const fp b96 = -107.0 / 9.0;
	static const fp b97 = 67.0 / 90.0;
	static const fp b98 = 3.0;
	static const fp b10_1 = -91.0 / 108.0;
	static const fp b10_4 = 23.0 / 108.0;
	static const fp b10_5 = -976.0 / 135.0;
	static const fp b10_6 = 311.0 / 54.0;
	static const fp b10_7 = -19.0 / 60.0;
	static const fp b10_8 = 17.0 / 6.0;
	static const fp b10_9 = -1.0 / 12.0;
	static const fp b11_1 = 2383.0 / 4100.0;
	static const fp b11_4 = -341.0 / 164.0;
	static const fp b11_5 = 4496.0 / 1025.0;
	static const fp b11_6 = -301.0 / 82.0;
	static const fp b11_7 = 2133.0 / 4100.0;
	static const fp b11_8 = 45.0 / 82.0;
	static const fp b11_9 = 45.0 / 164.0;
	static const fp b11_10 = 18.0 / 41.0;
	static const fp b12_1 = 3.0 / 205.0;
	static const fp b12_6 = - 6.0 / 41.0;
	static const fp b12_7 = - 3.0 / 205.0;
	static const fp b12_8 = - 3.0 / 41.0;
	static const fp b12_9 = 3.0 / 41.0;
	static const fp b12_10 = 6.0 / 41.0;
	static const fp b13_1 = -1777.0 / 4100.0;
	static const fp b13_4 = -341.0 / 164.0;
	static const fp b13_5 = 4496.0 / 1025.0;
	static const fp b13_6 = -289.0 / 82.0;
	static const fp b13_7 = 2193.0 / 4100.0;
	static const fp b13_8 = 51.0 / 82.0;
	static const fp b13_9 = 33.0 / 164.0;
	static const fp b13_10 = 12.0 / 41.0;

	static const fp err_factor  = -41.0 / 840.0;

	fp h2_7 = a2 * h;

	fp timeinst_temp;
	fp* initvalu_temp;
	fp** finavalu_temp;

	int i;

	//======================================================================================================================================================
	//		TEMPORARY STORAGE ALLOCATION
	//======================================================================================================================================================

	initvalu_temp= (fp *) malloc(EQUATIONS* sizeof(fp));

	finavalu_temp= (fp **) malloc(13* sizeof(fp *));
	for (i= 0; i<13; i++){
		finavalu_temp[i]= (fp *) malloc(EQUATIONS* sizeof(fp));
	}

	//======================================================================================================================================================
	//		EVALUATIONS	[UNROLLED LOOP] [SEQUENTIAL DEPENDENCY]
	//======================================================================================================================================================

	//===================================================================================================
	//		1
	//===================================================================================================

	timeinst_temp = timeinst;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] ;
		// printf("initvalu[%d] = %f\n", i, initvalu[i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[0],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		2
	//===================================================================================================

	timeinst_temp = timeinst+h2_7;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h2_7 * (finavalu_temp[0][i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[1],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,\

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		3
	//===================================================================================================

	timeinst_temp = timeinst+a3*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b31*finavalu_temp[0][i] + b32*finavalu_temp[1][i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[2],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		4
	//===================================================================================================

	timeinst_temp = timeinst+a4*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b41*finavalu_temp[0][i] + b43*finavalu_temp[2][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[3],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		5
	//===================================================================================================

	timeinst_temp = timeinst+a5*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b51*finavalu_temp[0][i] + b53*finavalu_temp[2][i] + b54*finavalu_temp[3][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[4],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		6
	//===================================================================================================

	timeinst_temp = timeinst+a6*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b61*finavalu_temp[0][i] + b64*finavalu_temp[3][i] + b65*finavalu_temp[4][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[5],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		7
	//===================================================================================================

	timeinst_temp = timeinst+a7*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b71*finavalu_temp[0][i] + b74*finavalu_temp[3][i] + b75*finavalu_temp[4][i] + b76*finavalu_temp[5][i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[6],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		8
	//===================================================================================================

	timeinst_temp = timeinst+a8*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b81*finavalu_temp[0][i] + b85*finavalu_temp[4][i] + b86*finavalu_temp[5][i] + b87*finavalu_temp[6][i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[7],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		9
	//===================================================================================================

	timeinst_temp = timeinst+a9*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b91*finavalu_temp[0][i] + b94*finavalu_temp[3][i] + b95*finavalu_temp[4][i] + b96*finavalu_temp[5][i] + b97*finavalu_temp[6][i]+ b98*finavalu_temp[7][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[8],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		10
	//===================================================================================================

	timeinst_temp = timeinst+a10*h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b10_1*finavalu_temp[0][i] + b10_4*finavalu_temp[3][i] + b10_5*finavalu_temp[4][i] + b10_6*finavalu_temp[5][i] + b10_7*finavalu_temp[6][i] + b10_8*finavalu_temp[7][i] + b10_9*finavalu_temp[8] [i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[9],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		11
	//===================================================================================================

	timeinst_temp = timeinst+h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b11_1*finavalu_temp[0][i] + b11_4*finavalu_temp[3][i] + b11_5*finavalu_temp[4][i] + b11_6*finavalu_temp[5][i] + b11_7*finavalu_temp[6][i] + b11_8*finavalu_temp[7][i] + b11_9*finavalu_temp[8][i]+ b11_10 * finavalu_temp[9][i]);
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[10],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		12
	//===================================================================================================

	timeinst_temp = timeinst;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b12_1*finavalu_temp[0][i] + b12_6*finavalu_temp[5][i] + b12_7*finavalu_temp[6][i] + b12_8*finavalu_temp[7][i] + b12_9*finavalu_temp[8][i] + b12_10 * finavalu_temp[9][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[11],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//===================================================================================================
	//		13
	//===================================================================================================

	timeinst_temp = timeinst+h;
	for(i=0; i<EQUATIONS; i++){
		initvalu_temp[i] = initvalu[i] + h * ( b13_1*finavalu_temp[0][i] + b13_4*finavalu_temp[3][i] + b13_5*finavalu_temp[4][i] + b13_6*finavalu_temp[5][i] + b13_7*finavalu_temp[6][i] + b13_8*finavalu_temp[7][i] + b13_9*finavalu_temp[8][i] + b13_10*finavalu_temp[9][i] + finavalu_temp[11][i]) ;
	}

	master(	timeinst_temp,
					initvalu_temp,
					parameter,
					finavalu_temp[12],
					com,

					d_initvalu,
					d_finavalu,
					d_params,
					d_com,

					command_queue,
					kernel,

					timecopyin,
					timecopykernel,
					timecopyout);

	//======================================================================================================================================================
	//		FINAL VALUE
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		finavalu[i]= initvalu[i] +  h * (c_1_11 * (finavalu_temp[0][i] + finavalu_temp[10][i])  + c6 * finavalu_temp[5][i] + c_7_8 * (finavalu_temp[6][i] + finavalu_temp[7][i]) + c_9_10 * (finavalu_temp[8][i] + finavalu_temp[9][i]) );
	}

	//======================================================================================================================================================
	//		RETURN
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		error[i] = fabs(err_factor * (finavalu_temp[0][i] + finavalu_temp[10][i] - finavalu_temp[11][i] - finavalu_temp[12][i]));
	}

	//======================================================================================================================================================
	//		DEALLOCATION
	//======================================================================================================================================================

	free(initvalu_temp);
	free(finavalu_temp);

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
