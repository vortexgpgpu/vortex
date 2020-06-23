/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef LBM_KERNEL_CL
#define LBM_KERNEL_CL


/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#ifndef _LAYOUT_CONFIG_H_
#define _LAYOUT_CONFIG_H_

/*############################################################################*/

//Unchangeable settings: volume simulation size for the given example
#define SIZE_X (32)
#define SIZE_Y (32)
#define SIZE_Z (32)

//Changeable settings
//Padding in each dimension
#define PADDING_X (8)
#define PADDING_Y (0)
#define PADDING_Z (4)

//Pitch in each dimension
#define PADDED_X (SIZE_X+PADDING_X)
#define PADDED_Y (SIZE_Y+PADDING_Y)
#define PADDED_Z (SIZE_Z+PADDING_Z)

#define TOTAL_CELLS (SIZE_X*SIZE_Y*SIZE_Z)
#define TOTAL_PADDED_CELLS (PADDED_X*PADDED_Y*PADDED_Z)

//Flattening function
//  This macro will be used to map a 3-D index and element to a value
//  The macro below implements the equivalent of a 3-D array of 
//  20-element structures in C standard layout.
#define CALC_INDEX(x,y,z,e) ( e + N_CELL_ENTRIES*\
                               ((x)+(y)*PADDED_X+(z)*PADDED_X*PADDED_Y) )

#define MARGIN (CALC_INDEX(0, 0, 2, 0) - CALC_INDEX(0,0,0,0))

// Set this value to 1 for GATHER, or 0 for SCATTER
#if 1
#define GATHER
#else
#define SCATTER
#endif

//OpenCL block size (not trivially changeable here)
#define BLOCK_SIZE SIZE_X

/*############################################################################*/

typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;

#define N_DISTR_FUNCS FLAGS

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;

#endif /* _CONFIG_H_ */


#ifndef _LBM_MARCOS_H
#define _LBM_MACROS_H_

#define OMEGA (1.95f)

#define OUTPUT_PRECISION float

#define BOOL int
#define TRUE (-1)
#define FALSE (0)

#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

/*############################################################################*/

typedef float* LBM_Grid;//float LBM_Grid[PADDED_Z*PADDED_Y*PADDED_X*N_CELL_ENTRIES];
typedef LBM_Grid* LBM_GridPtr;

/*############################################################################*/


#define SWEEP_X  __temp_x__
#define SWEEP_Y  __temp_y__
#define SWEEP_Z  __temp_z__
#define SWEEP_VAR int __temp_x__, __temp_y__, __temp_z__;

#define SWEEP_START(x1,y1,z1,x2,y2,z2) \
	for( __temp_z__ = z1; \
	     __temp_z__ < z2; \
		__temp_z__++) { \
            for( __temp_y__ = 0; \
                 __temp_y__ < SIZE_Y; \
                 __temp_y__++) { \
		for(__temp_x__ = 0; \
	            __temp_x__ < SIZE_X; \
                    __temp_x__++) { \

#define SWEEP_END }}}


#define GRID_ENTRY(g,x,y,z,e)          ((g)[CALC_INDEX( x,  y,  z, e)])
#define GRID_ENTRY_SWEEP(g,dx,dy,dz,e) ((g)[CALC_INDEX((dx)+SWEEP_X, (dy)+SWEEP_Y, (dz)+SWEEP_Z, e)])

#define LOCAL(g,e)       (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_C(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_N(g,e)  (GRID_ENTRY_SWEEP( g,  0, +1,  0, e ))
#define NEIGHBOR_S(g,e)  (GRID_ENTRY_SWEEP( g,  0, -1,  0, e ))
#define NEIGHBOR_E(g,e)  (GRID_ENTRY_SWEEP( g, +1,  0,  0, e ))
#define NEIGHBOR_W(g,e)  (GRID_ENTRY_SWEEP( g, -1,  0,  0, e ))
#define NEIGHBOR_T(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, +1, e ))
#define NEIGHBOR_B(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, -1, e ))
#define NEIGHBOR_NE(g,e) (GRID_ENTRY_SWEEP( g, +1, +1,  0, e ))
#define NEIGHBOR_NW(g,e) (GRID_ENTRY_SWEEP( g, -1, +1,  0, e ))
#define NEIGHBOR_SE(g,e) (GRID_ENTRY_SWEEP( g, +1, -1,  0, e ))
#define NEIGHBOR_SW(g,e) (GRID_ENTRY_SWEEP( g, -1, -1,  0, e ))
#define NEIGHBOR_NT(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, +1, e ))
#define NEIGHBOR_NB(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, -1, e ))
#define NEIGHBOR_ST(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, +1, e ))
#define NEIGHBOR_SB(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, -1, e ))
#define NEIGHBOR_ET(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, +1, e ))
#define NEIGHBOR_EB(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, -1, e ))
#define NEIGHBOR_WT(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, +1, e ))
#define NEIGHBOR_WB(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, -1, e ))


#ifdef SCATTER

#define SRC_C(g)  (LOCAL( g, C  ))
#define SRC_N(g)  (LOCAL( g, N  ))
#define SRC_S(g)  (LOCAL( g, S  ))
#define SRC_E(g)  (LOCAL( g, E  ))
#define SRC_W(g)  (LOCAL( g, W  ))
#define SRC_T(g)  (LOCAL( g, T  ))
#define SRC_B(g)  (LOCAL( g, B  ))
#define SRC_NE(g) (LOCAL( g, NE ))
#define SRC_NW(g) (LOCAL( g, NW ))
#define SRC_SE(g) (LOCAL( g, SE ))
#define SRC_SW(g) (LOCAL( g, SW ))
#define SRC_NT(g) (LOCAL( g, NT ))
#define SRC_NB(g) (LOCAL( g, NB ))
#define SRC_ST(g) (LOCAL( g, ST ))
#define SRC_SB(g) (LOCAL( g, SB ))
#define SRC_ET(g) (LOCAL( g, ET ))
#define SRC_EB(g) (LOCAL( g, EB ))
#define SRC_WT(g) (LOCAL( g, WT ))
#define SRC_WB(g) (LOCAL( g, WB ))

#define DST_C(g)  (NEIGHBOR_C ( g, C  ))
#define DST_N(g)  (NEIGHBOR_N ( g, N  ))
#define DST_S(g)  (NEIGHBOR_S ( g, S  ))
#define DST_E(g)  (NEIGHBOR_E ( g, E  ))
#define DST_W(g)  (NEIGHBOR_W ( g, W  ))
#define DST_T(g)  (NEIGHBOR_T ( g, T  ))
#define DST_B(g)  (NEIGHBOR_B ( g, B  ))
#define DST_NE(g) (NEIGHBOR_NE( g, NE ))
#define DST_NW(g) (NEIGHBOR_NW( g, NW ))
#define DST_SE(g) (NEIGHBOR_SE( g, SE ))
#define DST_SW(g) (NEIGHBOR_SW( g, SW ))
#define DST_NT(g) (NEIGHBOR_NT( g, NT ))
#define DST_NB(g) (NEIGHBOR_NB( g, NB ))
#define DST_ST(g) (NEIGHBOR_ST( g, ST ))
#define DST_SB(g) (NEIGHBOR_SB( g, SB ))
#define DST_ET(g) (NEIGHBOR_ET( g, ET ))
#define DST_EB(g) (NEIGHBOR_EB( g, EB ))
#define DST_WT(g) (NEIGHBOR_WT( g, WT ))
#define DST_WB(g) (NEIGHBOR_WB( g, WB ))

#else /* GATHER */

#define SRC_C(g)  (NEIGHBOR_C ( g, C  ))
#define SRC_N(g)  (NEIGHBOR_S ( g, N  ))
#define SRC_S(g)  (NEIGHBOR_N ( g, S  ))
#define SRC_E(g)  (NEIGHBOR_W ( g, E  ))
#define SRC_W(g)  (NEIGHBOR_E ( g, W  ))
#define SRC_T(g)  (NEIGHBOR_B ( g, T  ))
#define SRC_B(g)  (NEIGHBOR_T ( g, B  ))
#define SRC_NE(g) (NEIGHBOR_SW( g, NE ))
#define SRC_NW(g) (NEIGHBOR_SE( g, NW ))
#define SRC_SE(g) (NEIGHBOR_NW( g, SE ))
#define SRC_SW(g) (NEIGHBOR_NE( g, SW ))
#define SRC_NT(g) (NEIGHBOR_SB( g, NT ))
#define SRC_NB(g) (NEIGHBOR_ST( g, NB ))
#define SRC_ST(g) (NEIGHBOR_NB( g, ST ))
#define SRC_SB(g) (NEIGHBOR_NT( g, SB ))
#define SRC_ET(g) (NEIGHBOR_WB( g, ET ))
#define SRC_EB(g) (NEIGHBOR_WT( g, EB ))
#define SRC_WT(g) (NEIGHBOR_EB( g, WT ))
#define SRC_WB(g) (NEIGHBOR_ET( g, WB ))

#define DST_C(g)  (LOCAL( g, C  ))
#define DST_N(g)  (LOCAL( g, N  ))
#define DST_S(g)  (LOCAL( g, S  ))
#define DST_E(g)  (LOCAL( g, E  ))
#define DST_W(g)  (LOCAL( g, W  ))
#define DST_T(g)  (LOCAL( g, T  ))
#define DST_B(g)  (LOCAL( g, B  ))
#define DST_NE(g) (LOCAL( g, NE ))
#define DST_NW(g) (LOCAL( g, NW ))
#define DST_SE(g) (LOCAL( g, SE ))
#define DST_SW(g) (LOCAL( g, SW ))
#define DST_NT(g) (LOCAL( g, NT ))
#define DST_NB(g) (LOCAL( g, NB ))
#define DST_ST(g) (LOCAL( g, ST ))
#define DST_SB(g) (LOCAL( g, SB ))
#define DST_ET(g) (LOCAL( g, ET ))
#define DST_EB(g) (LOCAL( g, EB ))
#define DST_WT(g) (LOCAL( g, WT ))
#define DST_WB(g) (LOCAL( g, WB ))

#endif /* GATHER */

#define MAGIC_CAST(v) ((unsigned int*) ((void*) (&(v))))
#define FLAG_VAR(v) unsigned int* _aux_ = MAGIC_CAST(v)

#define TEST_FLAG_SWEEP(g,f)     ((*MAGIC_CAST(LOCAL(g, FLAGS))) & (f))
#define SET_FLAG_SWEEP(g,f)      {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG_SWEEP(g,f)    {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS_SWEEP(g) {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_)  =    0;}

#define TEST_FLAG(g,x,y,z,f)     ((*MAGIC_CAST(GRID_ENTRY(g, x, y, z, FLAGS))) & (f))
#define SET_FLAG(g,x,y,z,f)      {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG(g,x,y,z,f)    {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS(g,x,y,z) {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_)  =    0;}

/*############################################################################*/

#endif /* _CONFIG_H_ */


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
