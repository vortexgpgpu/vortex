//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "main.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL
//========================================================================================================================================================================================================200

__kernel void 
kernel_gpu_opencl(	// structures
					params_common d_common,					// 0

					// common_change
					__global fp* d_frame,					// 1	INPUT
					int d_frame_no,							// 2	INPUT

					// common
					__global int* d_endoRow,				// 3	INPUT
					__global int* d_endoCol,				// 4	INPUT
					__global int* d_tEndoRowLoc,			// 5	OUTPUT	common.endoPoints * common.no_frames
					__global int* d_tEndoColLoc,			// 6	OUTPUT	common.endoPoints * common.no_frames
					__global int* d_epiRow,					// 7	INPUT
					__global int* d_epiCol,					// 8	INPUT
					__global int* d_tEpiRowLoc,				// 9	OUTPUT	common.epiPoints * common.no_frames
					__global int* d_tEpiColLoc,				// 10	OUTPUT	common.epiPoints * common.no_frames

					// common_unique
					__global fp* d_endoT,					// 11	OUTPUT	common.in_elem * common.endoPoints
					__global fp* d_epiT,					// 12	OUTPUT	common.in_elem * common.epiPoints
					__global fp* d_in2_all,					// 13	OUTPUT	common.in2_elem * common.allPoints
					__global fp* d_conv_all,				// 14	OUTPUT	common.conv_elem * common.allPoints
					__global fp* d_in2_pad_cumv_all,		// 15	OUTPUT	common.in2_pad_cumv_elem * common.allPoints
					__global fp* d_in2_pad_cumv_sel_all,	// 16	OUTPUT	common.in2_pad_cumv_sel_elem * common.allPoints
					__global fp* d_in2_sub_cumh_all,		// 17	OUTPUT	common.in2_sub_cumh_elem * common.allPoints
					__global fp* d_in2_sub_cumh_sel_all,	// 18	OUTPUT	common.in2_sub_cumh_sel_elem * common.allPoints
					__global fp* d_in2_sub2_all,			// 19	OUTPUT	common.in2_sub2_elem * common.allPoints
					__global fp* d_in2_sqr_all,				// 20	OUTPUT	common.in2_elem * common.allPoints
					__global fp* d_in2_sqr_sub2_all,		// 21	OUTPUT	common.in2_sub2_elem * common.allPoints
					__global fp* d_in_sqr_all,				// 22	OUTPUT	common.in_elem * common.allPoints
					__global fp* d_tMask_all,				// 23	OUTPUT	common.tMask_elem * common.allPoints
					__global fp* d_mask_conv_all,			// 24	OUTPUT	common.mask_conv_elem * common.allPoints

					// // local
					// __local fp* d_in_mod_temp,			// 25	OUTPUT	common.in_elem
					// __local fp* in_partial_sum,			// 26	OUTPUT	common.in_cols
					// __local fp* in_sqr_partial_sum,		// 27	OUTPUT	common.in_sqr_rows
					// __local fp* par_max_val,				// 28	OUTPUT	common.mask_conv_rows
					// __local int* par_max_coo)			// 29	OUTPUT	common.mask_conv_rows

					// local
					__global fp* d_in_mod_temp_all,			// 25	OUTPUT	common.in_elem * common.allPoints
					__global fp* in_partial_sum_all,		// 26	OUTPUT	common.in_cols * common.allPoints
					__global fp* in_sqr_partial_sum_all,	// 27	OUTPUT	common.in_sqr_rows * common.allPoints
					__global fp* par_max_val_all,			// 28	OUTPUT	common.mask_conv_rows * common.allPoints
					__global int* par_max_coo_all,			// 29	OUTPUT	common.mask_conv_rows * common.allPoints

					__global fp* in_final_sum_all,			// 30	OUTPUT	common.allPoints
					__global fp* in_sqr_final_sum_all,		// 31	OUTPUT	common.allPoints
					__global fp* denomT_all,				// 32	OUTPUT	common.allPoints

					__global fp* checksum)					// 33	OUTPUT	100

{

	//======================================================================================================================================================150
	//	COMMON VARIABLES
	//======================================================================================================================================================150

	// __global fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	fp s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	fp sum;
	int pos_ori;
	fp temp;
	fp temp2;
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp mean;
	fp mean_sqr;
	fp variance;
	fp deviation;
	int pointer;
	int ori_pointer;
	int loc_pointer;

	// __local fp in_final_sum;
	// __local fp in_sqr_final_sum;
	// __local fp denomT;

	//======================================================================================================================================================150
	//	BLOCK/THREAD IDs
	//======================================================================================================================================================150

	int bx = get_group_id(0);															// get current horizontal block index (0-n)
	int tx = get_local_id(0);															// get current horizontal thread index (0-n)
	int ei_new;

	//======================================================================================================================================================150
	//	UNIQUE STRUCTURE RECONSTRUCTED HERE
	//======================================================================================================================================================150

	// common
	__global fp* d_common_change_d_frame = &d_frame[0];

	// offsets for either endo or epi points (separate arrays for endo and epi points)
	int d_unique_point_no;
	__global int* d_unique_d_Row;
	__global int* d_unique_d_Col;
	__global int* d_unique_d_tRowLoc;
	__global int* d_unique_d_tColLoc;
	__global fp* d_in;
	if(bx < d_common.endoPoints){
		d_unique_point_no = bx;													// endo point number 0-???
		d_unique_d_Row = d_endoRow;												// initial endo row coordinates
		d_unique_d_Col = d_endoCol;												// initial endo col coordinates
		d_unique_d_tRowLoc = d_tEndoRowLoc;										// all endo row coordinates
		d_unique_d_tColLoc = d_tEndoColLoc;										// all endo col coordinates
		d_in = &d_endoT[d_unique_point_no * d_common.in_elem];					// endo templates
	}
	else{
		d_unique_point_no = bx-d_common.endoPoints;								// epi point number 0-???
		d_unique_d_Row = d_epiRow;												// initial epi row coordinates
		d_unique_d_Col = d_epiCol;												// initial epi col coordinates
		d_unique_d_tRowLoc = d_tEpiRowLoc;										// all epi row coordinates
		d_unique_d_tColLoc = d_tEpiColLoc;										// all epi col coordinates
		d_in = &d_epiT[d_unique_point_no * d_common.in_elem];					// epi templates
	}

	// offsets for all points (one array for all points)
	__global fp* d_unique_d_in2 = &d_in2_all[bx*d_common.in2_elem];
	__global fp* d_unique_d_conv = &d_conv_all[bx*d_common.conv_elem];
	__global fp* d_unique_d_in2_pad_cumv = &d_in2_pad_cumv_all[bx*d_common.in2_pad_cumv_elem];
	__global fp* d_unique_d_in2_pad_cumv_sel = &d_in2_pad_cumv_sel_all[bx*d_common.in2_pad_cumv_sel_elem];
	__global fp* d_unique_d_in2_sub_cumh = &d_in2_sub_cumh_all[bx*d_common.in2_sub_cumh_elem];
	__global fp* d_unique_d_in2_sub_cumh_sel = &d_in2_sub_cumh_sel_all[bx*d_common.in2_sub_cumh_sel_elem];
	__global fp* d_unique_d_in2_sub2 = &d_in2_sub2_all[bx*d_common.in2_sub2_elem];
	__global fp* d_unique_d_in2_sqr = &d_in2_sqr_all[bx*d_common.in2_sqr_elem];
	__global fp* d_unique_d_in2_sqr_sub2 = &d_in2_sqr_sub2_all[bx*d_common.in2_sqr_sub2_elem];
	__global fp* d_unique_d_in_sqr = &d_in_sqr_all[bx*d_common.in_sqr_elem];
	__global fp* d_unique_d_tMask = &d_tMask_all[bx*d_common.tMask_elem];
	__global fp* d_unique_d_mask_conv = &d_mask_conv_all[bx*d_common.mask_conv_elem];

	// used to be local
	__global fp* d_in_mod_temp = &d_in_mod_temp_all[bx*d_common.in_elem];
	__global fp* in_partial_sum = &in_partial_sum_all[bx*d_common.in_cols];
	__global fp* in_sqr_partial_sum = &in_sqr_partial_sum_all[bx*d_common.in_sqr_rows];
	__global fp* par_max_val = &par_max_val_all[bx*d_common.mask_conv_rows];
	__global int* par_max_coo = &par_max_coo_all[bx*d_common.mask_conv_rows];

	__global fp* in_final_sum = &in_final_sum_all[bx];
	__global fp* in_sqr_final_sum = &in_sqr_final_sum_all[bx];
	__global fp* denomT = &denomT_all[bx];

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	Initialize checksum
	//======================================================================================================================================================150
#ifdef TEST_CHECKSUM
	if(bx==0 && tx==0){

		for(i=0; i<CHECK; i++){
			checksum[i] = 0;
		}

	}
#endif
	//======================================================================================================================================================150
	//	INITIAL COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	// generate templates based on the first frame only
	if(d_frame_no == 0){

		//====================================================================================================100
		//	Initialize cross-frame variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block and for all frames
			for(i=0; i<d_common.no_frames; i++){
				d_unique_d_tRowLoc[d_unique_point_no*d_common.no_frames+i] = 0;
				d_unique_d_tColLoc[d_unique_point_no*d_common.no_frames+i] = 0;
			}

			// this block
			for(i=0; i<d_common.in_elem; i++){
				d_in[i] = 0;
			}

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	UPDATE ROW LOC AND COL LOC
		//====================================================================================================100

		// uptade temporary endo/epi row/col coordinates (in each block corresponding to point, narrow work to one thread)
		ei_new = tx;
		if(ei_new == 0){

			// update temporary row/col coordinates
			pointer = d_unique_point_no*d_common.no_frames+d_frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no];
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no];

		}

		//====================================================================================================100
		//	CREATE TEMPLATES
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[col*d_common.in_rows+row] = d_common_change_d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	CHECKSUM
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[0] = checksum[0]+d_in[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PROCESS POINTS
	//======================================================================================================================================================150

	// process points in all frames except for the first one
	if(d_frame_no != 0){

		//====================================================================================================100
		//	Initialize frame-specific variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block
			for(i=0; i<d_common.in2_elem; i++){
				d_unique_d_in2[i] = 0;
			}
			for(i=0; i<d_common.conv_elem; i++){
				d_unique_d_conv[i] = 0;
			}
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				d_unique_d_in2_pad_cumv[i] = 0;
			}
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				d_unique_d_in2_pad_cumv_sel[i] = 0;
			}
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				d_unique_d_in2_sub_cumh[i] = 0;
			}
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				d_unique_d_in2_sub_cumh_sel[i] = 0;
			}
			for(i=0; i<d_common.in2_sub2_elem; i++){
				d_unique_d_in2_sub2[i] = 0;
			}
			for(i=0; i<d_common.in2_sqr_elem; i++){
				d_unique_d_in2_sqr[i] = 0;
			}
			for(i=0; i<d_common.in2_sqr_sub2_elem; i++){
				d_unique_d_in2_sqr_sub2[i] = 0;
			}
			for(i=0; i<d_common.in_sqr_elem; i++){
				d_unique_d_in_sqr[i] = 0;
			}
			for(i=0; i<d_common.tMask_elem; i++){
				d_unique_d_tMask[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_elem; i++){
				d_unique_d_mask_conv[i] = 0;
			}

			for(i=0; i<d_common.in_elem; i++){
				d_in_mod_temp[i] = 0;
			}
			for(i=0; i<d_common.in_cols; i++){
				in_partial_sum[i] = 0;
			}
			for(i=0; i<d_common.in_sqr_rows; i++){
				in_sqr_partial_sum[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_rows; i++){
				par_max_val[i] = 0;
			}
			for(i=0; i<d_common.mask_conv_rows; i++){
				par_max_coo[i] = 0;
			}

			in_final_sum[0] = 0;
			in_sqr_final_sum[0] = 0;
			denomT[0] = 0;

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	SELECTION
		//====================================================================================================100

		in2_rowlow = d_unique_d_Row[d_unique_point_no] - d_common.sSize;													// (1 to n+1)
		in2_collow = d_unique_d_Col[d_unique_point_no] - d_common.sSize;

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_rows == 0){
				row = d_common.in2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_unique_d_in2[ei_new] = d_common_change_d_frame[ori_col*d_common.frame_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		//	CHECKSUM
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_elem; i++){
				checksum[1] = checksum[1]+d_unique_d_in2[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//====================================================================================================100
		//	CONVOLUTION
		//====================================================================================================100

		//==================================================50
		//	ROTATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){
		// while(ei_new < 1){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// execution
			rot_row = (d_common.in_rows-1) - row;
			rot_col = (d_common.in_rows-1) - col;
			d_in_mod_temp[ei_new] = d_in[rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = d_in[0];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_in_pointer];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * d_common.in_elem];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * 2601];
			// if((d_unique_point_no * d_common.in_elem) > (2601*51
				// printf("frame_no IS %d\n", d_common_change[0].frame_no);
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no];

			// d_in_mod_temp[ei_new] = 1;
			// kot = d_in[rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = kot;
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_in_pointer+rot_col*d_common.in_rows+rot_row];
			// d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no * d_common.in_elem+rot_col*d_common.in_rows+rot_row];
			//d_in_mod_temp[ei_new] = d_unique_d_T[d_unique_point_no];
			// d_unique_d_T[d_unique_in_pointer+rot_col*d_common.in_rows+rot_row] = 1;
			// d_unique_d_T[d_unique_in_pointer] = 1;
			// d_endoT[d_unique_in_pointer] = 1;
			// d_in_mod_temp[ei_new] = 1;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[2] = checksum[2]+d_in_mod_temp[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	ACTUAL CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.conv_rows == 0){
				ic = d_common.conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.joffset;
			jp1 = j + 1;
			if(d_common.in2_cols < jp1){
				ja1 = jp1 - d_common.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.in_cols < j){
				ja2 = d_common.in_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.ioffset;
			ip1 = i + 1;
			
			if(d_common.in2_rows < ip1){
				ia1 = ip1 - d_common.in2_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.in_rows < i){
				ia2 = d_common.in_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_in_mod_temp[d_common.in_rows*(ja-1)+ia-1] * d_unique_d_in2[d_common.in2_rows*(jb-1)+ib-1];
				}
			}

			//d_unique_d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_conv[ei_new] = s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[3] = checksum[3]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		// 	CUMULATIVE SUM	(LOCAL)
		//====================================================================================================100

		//==================================================50
		//	PADD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2[ori_col*d_common.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[4] = checksum[4]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_cols; i++){
				checksum[5] = checksum[5]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[6] = checksum[6]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[7] = checksum[7]+d_unique_d_in2_sub_cumh[i];
			}
		}
#endif
		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[8] = checksum[8]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[9] = checksum[9]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[10] = checksum[10]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[11] = checksum[11]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[12] = checksum[12]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	CUMULATIVE SUM 2
		//====================================================================================================100

		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sqr_elem){

			temp = d_unique_d_in2[ei_new];
			d_unique_d_in2_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sqr_elem; i++){
				checksum[13] = checksum[13]+d_unique_d_in2_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//==================================================50

		//==================================================50
		//	PAD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_sqr_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_sqr_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2_sqr[ori_col*d_common.in2_sqr_rows+ori_row];
			}
			else{																							// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[14] = checksum[14]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[15] = checksum[15]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[16] = checksum[16]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[17] = checksum[17]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[18] = checksum[18]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_rows; i++){
				checksum[19] = checksum[19]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[20] = checksum[20]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[21] = checksum[21]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[22] = checksum[22]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	FINAL
		//====================================================================================================100

		//==================================================50
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			temp = d_unique_d_in2_sub2[ei_new];
			temp2 = d_unique_d_in2_sqr_sub2[ei_new] - (temp * temp / d_common.in_elem);
			if(temp2 < 0){
				temp2 = 0;
			}
			d_unique_d_in2_sqr_sub2[ei_new] = sqrt(temp2);
			

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[23] = checksum[23]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_sqr_elem){

			temp = d_in[ei_new];
			d_unique_d_in_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_elem; i++){
				checksum[24] = checksum[24]+d_unique_d_in_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	IN SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_cols){

			sum = 0;
			for(i = 0; i < d_common.in_rows; i++){

				sum = sum + d_in[ei_new*d_common.in_rows+i];

			}
			in_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_cols; i++){
				checksum[25] = checksum[25]+in_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	IN_SQR SUM
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.in_sqr_rows){
				
			sum = 0;
			for(i = 0; i < d_common.in_sqr_cols; i++){

				sum = sum + d_unique_d_in_sqr[ei_new+d_common.in_sqr_rows*i];

			}
			in_sqr_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_rows; i++){
				checksum[26] = checksum[26]+in_sqr_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	FINAL SUMMATION
		//==================================================50

		if(tx == 0){

			in_final_sum[0] = 0;
			for(i = 0; i<d_common.in_cols; i++){
				// in_final_sum = in_final_sum + in_partial_sum[i];
				in_final_sum[0] = in_final_sum[0] + in_partial_sum[i];
			}

		}else if(tx == 1){

			in_sqr_final_sum[0] = 0;
			for(i = 0; i<d_common.in_sqr_cols; i++){
				// in_sqr_final_sum = in_sqr_final_sum + in_sqr_partial_sum[i];
				in_sqr_final_sum[0] = in_sqr_final_sum[0] + in_sqr_partial_sum[i];
			}

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[27] = checksum[27]+in_final_sum[0]+in_sqr_final_sum[0];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	DENOMINATOR T
		//==================================================50

		if(tx == 0){

			// mean = in_final_sum / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean = in_final_sum[0] / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			// variance  = (in_sqr_final_sum / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			variance  = (in_sqr_final_sum[0] / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			// denomT = sqrt((float)(d_common.in_elem-1))*deviation;
			denomT[0] = sqrt((float)(d_common.in_elem-1))*deviation;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[28] = checksum[28]+denomT[i];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT[0];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[29] = checksum[29]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;
			d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum[0] / d_common.in_elem;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[30] = checksum[30]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_conv[ei_new] / d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}



		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[31] = checksum[31]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	TEMPLATE MASK CREATE
		//====================================================================================================100

		cent = d_common.sSize + d_common.tSize + 1;
		if(d_frame_no == 0){
			tMask_row = cent + d_unique_d_Row[d_unique_point_no] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_Col[d_unique_point_no] - d_unique_d_Col[d_unique_point_no] - 1;
		}
		else{
			pointer = d_unique_point_no*d_common.no_frames+d_frame_no-1;
			tMask_row = cent + d_unique_d_tRowLoc[pointer] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_tColLoc[pointer] - d_unique_d_Col[d_unique_point_no] - 1;
		}

		//work
		ei_new = tx;
		while(ei_new < d_common.tMask_elem){

			location = tMask_col*d_common.tMask_rows + tMask_row;

			if(ei_new==location){
				d_unique_d_tMask[ei_new] = 1;
			}
			else{
				d_unique_d_tMask[ei_new] = 0;
			}

			//go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.tMask_elem; i++){
				checksum[32] = checksum[32]+d_unique_d_tMask[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MASK CONVOLUTION
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.mask_conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.mask_conv_rows == 0){
				ic = d_common.mask_conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.mask_conv_joffset;
			jp1 = j + 1;
			if(d_common.mask_cols < jp1){
				ja1 = jp1 - d_common.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.tMask_cols < j){
				ja2 = d_common.tMask_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.mask_conv_ioffset;
			ip1 = i + 1;
			
			if(d_common.mask_rows < ip1){
				ia1 = ip1 - d_common.mask_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.tMask_rows < i){
				ia2 = d_common.tMask_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique_d_tMask[d_common.tMask_rows*(ja-1)+ia-1] * 1;
				}
			}

			// //d_unique_d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_mask_conv[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_elem; i++){
				checksum[33] = checksum[33]+d_unique_d_mask_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MAXIMUM VALUE
		//====================================================================================================100

		//==================================================50
		//	INITIAL SEARCH
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.mask_conv_rows){

			for(i=0; i<d_common.mask_conv_cols; i++){
				largest_coordinate_current = ei_new*d_common.mask_conv_rows+i;
				largest_value_current = fabs(d_unique_d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
			}
			par_max_coo[ei_new] = largest_coordinate;
			par_max_val[ei_new] = largest_value;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_rows; i++){
				checksum[34] = checksum[34]+par_max_coo[i]+par_max_val[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	FINAL SEARCH
		//==================================================50

		if(tx == 0){

			for(i = 0; i < d_common.mask_conv_rows; i++){
				if(par_max_val[i] > fin_max_val){
					fin_max_val = par_max_val[i];
					fin_max_coo = par_max_coo[i];
				}
			}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % d_common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / d_common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % d_common.mask_conv_rows == 0){
				largest_row = d_common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
			offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
			pointer = d_unique_point_no*d_common.no_frames+d_frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no] + offset_row;
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no] + offset_col;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[35] = checksum[35]+d_unique_d_tRowLoc[pointer]+d_unique_d_tColLoc[pointer];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PERIODIC COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	if(d_frame_no != 0 && (d_frame_no)%10 == 0){

		//====================================================================================================100
		// initialize cross-frame variables
		//====================================================================================================100
#ifdef INIT
		// only the first thread initializes
		if(tx==0){

			// this block
			for(i=0; i<d_common.in_elem; i++){
				d_in[i] = 0;
			}

		}
#endif
		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//====================================================================================================100
		// if the last frame in the bath, update template
		//====================================================================================================100

		// update coordinate
		loc_pointer = d_unique_point_no*d_common.no_frames+d_frame_no;

		d_unique_d_Row[d_unique_point_no] = d_unique_d_tRowLoc[loc_pointer];
		d_unique_d_Col[d_unique_point_no] = d_unique_d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = d_common.alpha*d_in[ei_new] + (1-d_common.alpha)*d_common_change_d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//==================================================50
		//	CHECKSUM
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[36] = checksum[36]+d_in[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	End
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
