#ifndef _CONVERT_DATASET_H
#define _CONVERT_DATASET_H

#ifdef __cplusplus
extern "C" {
#endif
int coo_to_jds(char* mtx_filename, int pad_rows, int warp_size, int pack_size,
	       int mirrored, int binary, int debug_level,
               float** data, int** data_row_ptr, int** nz_count, int** data_col_index,
               int** data_row_map, int* data_cols, int* dim, int* len, int* nz_count_len,
	       int* data_ptr_len);

#ifdef __cplusplus
}
#endif

#endif