// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DESCRIPTION
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	UPDATE
//======================================================================================================================================================150

//	Summary of changes by Lukasz G. Szafaryn:

//	1) The original code was obtained from: Mathematics Source Library (http://mymathlib.webtrellis.net/index.html)
//	2) This solver and particular solving algorithm used with it (embedded_fehlberg_7_8) were adapted to work with a set of equations, not just one like in original version.

//	3) In order for solver to provide deterministic number of steps (needed for particular amount of memore previousely allocated for results), every next step is incremented by 1 time unit (h_init).
//	4) Function assumes that time interval starts at 0 (xmin) and ends at integer value (xmax) specified by the uses as a parameter on command line.
//	5) The appropriate amount of memory is previousely allocated for that range (y).

//	5) This setup in 3) - 5) allows solver to adjust the step ony from current time instance to current time instance + 0.9. The next time instance is current time instance + 1;

//	6) Solver also takes parameters (params) that it then passes to the equations.

//	7) The original solver cannot handle cases when equations return NAN and INF values due to discontinuities and /0. That is why equations provided by user need to make sure that no NAN and INF are returned.

//	Last update: 15 DEC 09

//======================================================================================================================================================150
//	DESCRIPTION
//======================================================================================================================================================150

// int solver( fp (*f)(fp, fp), fp y[],        //
//       fp x, fp h, fp xmax, fp *h_next, fp tolerance )  //
//                                                                            //
//  Description:                                                              //
//     This function solves the differential equation y'=f(x,y) with the      //
//     initial condition y(x) = y[0].  The value at xmax is returned in y[1]. //
//     The function returns 0 if successful or -1 if it fails.                //
//                                                                            //
//  Arguments:                                                                //
//     fp *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y0) corresponding to the //
//                initial condition y(x0) = y0.                               //
//     fp y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at xmax.                               //
//     fp x   The initial value of x.                                     //
//     fp h   Initial step size.                                          //
//     fp xmax The endpoint of x.                                         //
//     fp *h_next   A pointer to the estimated step size for successive   //
//                      calls to solver.                       //
//     fp tolerance The tolerance of y(xmax), i.e. a solution is sought   //
//                so that the relative error < tolerance.                     //
//                                                                            //
//  Return Values:                                                            //
//     0   The solution of y' = f(x,y) from x to xmax is stored y[1] and      //
//         h_next has the value to the next size to try.                      //
//    -1   The solution of y' = f(x,y) from x to xmax failed.                 //
//    -2   Failed because either xmax < x or the step size h <= 0.            //
//    -3   Memory limit allocated for results was reached                     //

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "common.h"								// (in path provided here)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "embedded_fehlberg_7_8.h"				// (in path provided here)

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdlib.h>									// (in path known to compiler)	needed by malloc, free
#include <math.h>									// (in path known to compiler)	needed by pow, fabs

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	FUNCTION
//========================================================================================================================================================================================================200

int 
solver(	fp **y,
		fp *x,
		int xmax,
		fp *params,
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

	//========================================================================================================================
	//	VARIABLES
	//========================================================================================================================

	// solver parameters
	fp err_exponent;
	int error;
	int outside;
	fp h;
	fp h_init;
	fp tolerance;
	int xmin;

	// memory
	fp scale_min;
	fp scale_fina;
	fp* err= (fp *) malloc(EQUATIONS* sizeof(fp));
	fp* scale= (fp *) malloc(EQUATIONS* sizeof(fp));
	fp* yy= (fp *) malloc(EQUATIONS* sizeof(fp));

	// counters
	int i, j, k;

	//========================================================================================================================
	//		INITIAL SETUP
	//========================================================================================================================

	// solver parameters
	err_exponent = 1.0 / 7.0;
	h_init = 1;
	h = h_init;
	xmin = 0;
	tolerance = 10 / (fp)(xmax-xmin);

	// save value for initial time instance
	x[0] = 0;

	//========================================================================================================================
	//		CHECKING
	//========================================================================================================================

	// Verify that the step size is positive and that the upper endpoint of integration is greater than the initial enpoint.               //
	if (xmax < xmin || h <= 0.0){
		return -2;
	}

	// If the upper endpoint of the independent variable agrees with the initial value of the independent variable.  Set the value of the dependent variable and return success. //
	if (xmax == xmin){
		return 0; 
	}

	// Insure that the step size h is not larger than the length of the integration interval.                                            //
	if (h > (xmax - xmin) ) { 
		h = (fp)xmax - (fp)xmin; 
	}

	//========================================================================================================================
	//		SOLVING
	//========================================================================================================================

	printf("Time Steps: ");
	fflush(0);

	for(k=1; k<=xmax; k++) {											// start after initial value

		x[k] = k-1;
		h = h_init;

		//==========================================================================================
		//		REINITIALIZE VARIABLES
		//==========================================================================================

		scale_fina = 1.0;

		//==========================================================================================
		//		MAKE ATTEMPTS TO MINIMIZE ERROR
		//==========================================================================================

		// make attempts to minimize error
		for (j = 0; j < ATTEMPTS; j++) {

			//============================================================
			//		REINITIALIZE VARIABLES
			//============================================================

			error = 0;
			outside = 0;
			scale_min = MAX_SCALE_FACTOR;

			//============================================================
			//		EVALUATE ALL EQUATIONS
			//============================================================

			embedded_fehlberg_7_8(	x[k],
									h,
									y[k-1],
									y[k],
									err,
									params,
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

			//============================================================
			//		IF THERE WAS NO ERROR FOR ANY OF EQUATIONS, SET SCALE AND LEAVE THE LOOP
			//============================================================

			for(i=0; i<EQUATIONS; i++){
				if(err[i] > 0){
					error = 1;
				}
			}
			if (error != 1) {
				scale_fina = MAX_SCALE_FACTOR; 
				break;
			}

			//============================================================
			//		FIGURE OUT SCALE AS THE MINIMUM OF COMPONENT SCALES
			//============================================================

			for(i=0; i<EQUATIONS; i++){
				if(y[k-1][i] == 0.0){
					yy[i] = tolerance;
				}
				else{
					yy[i] = fabs(y[k-1][i]);
				}
				scale[i] = 0.8 * pow( tolerance * yy[i] / err[i] , err_exponent );
				if(scale[i]<scale_min){
					scale_min = scale[i];
				}
			}
			scale_fina = min( max(scale_min,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

			//============================================================
			//		IF WITHIN TOLERANCE, FINISH ATTEMPTS...
			//============================================================

			for(i=0; i<EQUATIONS; i++){
				if ( err[i] > ( tolerance * yy[i] ) ){
					outside = 1;
				}
			}
			if (outside == 0){
				break;
			}

			//============================================================
			//		...OTHERWISE, ADJUST STEP FOR NEXT ATTEMPT
			//============================================================

			// scale next step in a default way
			h = h * scale_fina;

			// limit step to 0.9, because when it gets close to 1, it no longer makes sense, as 1 is already the next time instance (added to original algorithm)
			if (h >= 0.9) {
				h = 0.9;
			}

			// if instance+step exceeds range limit, limit to that range
			if ( x[k] + h > (fp)xmax ){
				h = (fp)xmax - x[k];
			}

			// if getting closer to range limit, decrease step
			else if ( x[k] + h + 0.5 * h > (fp)xmax ){
				h = 0.5 * h;
			}

		}

		//==========================================================================================
		//		SAVE TIME INSTANCE THAT SOLVER ENDED UP USING
		//==========================================================================================

		x[k] = x[k] + h;

		//==========================================================================================
		//		IF MAXIMUM NUMBER OF ATTEMPTS REACHED AND CANNOT GIVE SOLUTION, EXIT PROGRAM WITH ERROR
		//==========================================================================================

		if ( j >= ATTEMPTS ) {
			return -1; 
		}

		printf("%d ", k);
		fflush(0);

	}

	printf("\n");
	fflush(0);

	//========================================================================================================================
	//		FREE MEMORY
	//========================================================================================================================

	free(err);
	free(scale);
	free(yy);

	//========================================================================================================================
	//		FINAL RETURN
	//========================================================================================================================

	return 0;

//======================================================================================================================================================
//======================================================================================================================================================
//		END OF SOLVER FUNCTION
//======================================================================================================================================================
//======================================================================================================================================================

} 

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
