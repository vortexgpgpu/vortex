/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef LBM_KERNEL_CL
#define LBM_KERNEL_CL

#include "layout_config.h"
#include "lbm_macros.h"
/******************************************************************************/

__kernel void performStreamCollide_kernel( __global float* srcGrid, __global float* dstGrid )
{
	srcGrid += MARGIN;
	dstGrid += MARGIN;


	//Using some predefined macros here.  Consider this the declaration 
        //  and initialization of the variables SWEEP_X, SWEEP_Y and SWEEP_Z

        SWEEP_VAR
	SWEEP_X = get_local_id(0);
	SWEEP_Y = get_group_id(0);
	SWEEP_Z = get_group_id(1);
	
	float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
	float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
	float tempSB, tempET, tempEB, tempWT, tempWB ;

	//Load all of the input fields
	//This is a gather operation of the SCATTER preprocessor variable
        // is undefined in layout_config.h, or a "local" read otherwise
	tempC = SRC_C(srcGrid);

	tempN = SRC_N(srcGrid);
	tempS = SRC_S(srcGrid);
	tempE = SRC_E(srcGrid);
	tempW = SRC_W(srcGrid);
	tempT = SRC_T(srcGrid);
	tempB = SRC_B(srcGrid);

	tempNE = SRC_NE(srcGrid);
	tempNW = SRC_NW(srcGrid);
	tempSE = SRC_SE(srcGrid);
	tempSW = SRC_SW(srcGrid);
	tempNT = SRC_NT(srcGrid);
	tempNB = SRC_NB(srcGrid);
	tempST = SRC_ST(srcGrid);
	tempSB = SRC_SB(srcGrid);
	tempET = SRC_ET(srcGrid);
	tempEB = SRC_EB(srcGrid);
	tempWT = SRC_WT(srcGrid);
	tempWB = SRC_WB(srcGrid);

	//Test whether the cell is fluid or obstacle
	if(as_uint(LOCAL(srcGrid,FLAGS)) & (OBSTACLE)) {
		
		//Swizzle the inputs: reflect any fluid coming into this cell 
		// back to where it came from
		temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
		temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
		temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
		temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
		temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
		temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp; 
		temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
		temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
		temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
	}
	else {
 
                //The math meat of LBM: ignore for optimization
	        float ux, uy, uz, rho, u2;
		float temp1, temp2, temp_base;
		rho = tempC + tempN
			+ tempS + tempE
			+ tempW + tempT
			+ tempB + tempNE
			+ tempNW + tempSE
			+ tempSW + tempNT
			+ tempNB + tempST
			+ tempSB + tempET
			+ tempEB + tempWT
			+ tempWB;

		ux = + tempE - tempW
			+ tempNE - tempNW
			+ tempSE - tempSW
			+ tempET + tempEB
			- tempWT - tempWB;

		uy = + tempN - tempS
			+ tempNE + tempNW
			- tempSE - tempSW
			+ tempNT + tempNB
			- tempST - tempSB;

		uz = + tempT - tempB
			+ tempNT - tempNB
			+ tempST - tempSB
			+ tempET - tempEB
			+ tempWT - tempWB;		
		
		ux /= rho;
		uy /= rho;
		uz /= rho;

		if(as_uint(LOCAL(srcGrid,FLAGS)) & (ACCEL)) {

			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
		temp_base = OMEGA*rho;
		temp1 = DFL1*temp_base;

		//Put the output values for this cell in the shared memory
		temp_base = OMEGA*rho;
		temp1 = DFL1*temp_base;
		temp2 = 1.0f-OMEGA;
		tempC = temp2*tempC + temp1*(                                 - u2);
	        temp1 = DFL2*temp_base;	
		tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
		tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
		tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
		tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
		tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
		tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
		temp1 = DFL3*temp_base;
		tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	}

	//Write the results computed above
	//This is a scatter operation of the SCATTER preprocessor variable
        // is defined in layout_config.h, or a "local" write otherwise
	DST_C ( dstGrid ) = tempC;

	DST_N ( dstGrid ) = tempN; 
	DST_S ( dstGrid ) = tempS;
	DST_E ( dstGrid ) = tempE;
	DST_W ( dstGrid ) = tempW;
	DST_T ( dstGrid ) = tempT;
	DST_B ( dstGrid ) = tempB;

	DST_NE( dstGrid ) = tempNE;
	DST_NW( dstGrid ) = tempNW;
	DST_SE( dstGrid ) = tempSE;
	DST_SW( dstGrid ) = tempSW;
	DST_NT( dstGrid ) = tempNT;
	DST_NB( dstGrid ) = tempNB;
	DST_ST( dstGrid ) = tempST;
	DST_SB( dstGrid ) = tempSB;
	DST_ET( dstGrid ) = tempET;
	DST_EB( dstGrid ) = tempEB;
	DST_WT( dstGrid ) = tempWT;
	DST_WB( dstGrid ) = tempWB;
}

#endif // LBM_KERNEL_CL
