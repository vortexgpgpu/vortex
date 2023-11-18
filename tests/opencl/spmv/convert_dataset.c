/*
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY when reading and writing
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

#include "convert_dataset.h"
#include "mmio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _mat_entry {
  int row, col; /* i,j */
  float val;
} mat_entry;

typedef struct _row_stats { // stats on each row
  int index;
  int size;
  int start;
  int padding;
} row_stats;

int sort_rows(const void *a, const void *b) {
  return (((mat_entry *)a)->row - ((mat_entry *)b)->row);
}
int sort_cols(const void *a, const void *b) {
  return (((mat_entry *)a)->col - ((mat_entry *)b)->col);
}
/* sorts largest by size first */
int sort_stats(const void *a, const void *b) {
  return (((row_stats *)b)->size - ((row_stats *)a)->size);
}

/*
 * COO to JDS matrix conversion.
 *
 * Needs to output both column and row major JDS formats
 * with the minor unit padded to a multiple of `pad_minor`
 * and the major unit arranged into groups of `group_size`
 *
 * Major unit is col, minor is row. Each block is either a scalar or vec4
 *
 * Inputs:
 *   mtx_filename - the file in COO format
 *   pad_rows - multiple of packed groups to pad each row to
 *   warp_size - each group of `warp_size` cols is padded to the same amount
 *   pack_size - number of items to pack
 *   mirrored - is the input mtx file a symmetric matrix? The other half will be
 *   	filled in if this is =1
 *   binary - does the sparse matrix file have values in the format "%d %d"
 *   	or "%d %d %lg"?
 *   debug_level - 0 for no output, 1 for simple JDS data, 2 for visual grid
 * Outputs:
 *   data - the raw data, padded and grouped as requested
 *   data_row_ptr - pointer offset into the `data` output, referenced
 *      by the current row loop index
 *   nz_count - number of non-zero entries in each row
 *      indexed by col / warp_size
 *   data_col_index - corresponds to the col that the same
 *      array index in `data` is at
 *   data_row_map - JDS row to real row
 *   data_cols - number of columns the output JDS matrix has
 *   dim - dimensions of the input matrix
 *   data_ptr_len - size of data_row_ptr (maps to original `depth` var)
 */
int coo_to_jds(char *mtx_filename, int pad_rows, int warp_size, int pack_size,
               int mirrored, int binary, int debug_level, float **data,
               int **data_row_ptr, int **nz_count, int **data_col_index,
               int **data_row_map, int *data_cols, int *dim, int *len,
               int *nz_count_len, int *data_ptr_len) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int nz;
  int i;
  float *val;
  mat_entry *entries;
  row_stats *stats;
  int rows, cols;

  if ((f = fopen(mtx_filename, "r")) == NULL)
    exit(1);

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode)) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nz)) != 0)
    exit(1);
  *dim = rows;

  if (mirrored) {
    // max possible size, might be less because diagonal values aren't doubled
    entries = (mat_entry *)malloc(2 * nz * sizeof(mat_entry));
  } else {
    entries = (mat_entry *)malloc(nz * sizeof(mat_entry));
  }

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  int cur_i = 0; // to account for mirrored diagonal entries

  for (i = 0; i < nz; i++, cur_i++) {
    if (!binary) {
      fscanf(f, "%d %d %f\n", &entries[cur_i].row, &entries[cur_i].col,
             &entries[cur_i].val);
    } else {
      fscanf(f, "%d %d\n", &entries[cur_i].row, &entries[cur_i].col);
      entries[cur_i].val = 1.0;
    }
    entries[cur_i].row--;
    entries[cur_i].col--;
    // printf("%d,%d = %f\n", entries[cur_i].row, entries[cur_i].col,
    // entries[cur_i].val);
    if (mirrored) {
      // fill in mirrored diagonal
      if (entries[cur_i].row != entries[cur_i].col) { // not a diagonal value
        cur_i++;
        entries[cur_i].val = entries[cur_i - 1].val;
        entries[cur_i].col = entries[cur_i - 1].row;
        entries[cur_i].row = entries[cur_i - 1].col;
        // printf("%d,%d = %f\n", entries[cur_i].row, entries[cur_i].col,
        // entries[cur_i].val);
      }
    }
  }
  // set new non-zero count
  nz = cur_i;
  if (debug_level >= 1) {
    printf("Converting COO to JDS format (%dx%d)\n%d matrix entries, warp size "
           "= %d, "
           "row padding align = %d, pack size = %d\n\n",
           rows, cols, nz, warp_size, pad_rows, pack_size);
  }
  if (f != stdin)
    fclose(f);

  /*
   * Now we have an array of values in entries
   * Transform to padded JDS format  - sort by rows, then fubini
   */

  int irow, icol = 0, istart = 0;
  int total_size = 0;

  /* Loop through each entry to figure out padding, grouping that determine
   * final data array size
   *
   * First calculate stats for each row
   *
   * Collect stats using the major_stats typedef
   */

  qsort(entries, nz, sizeof(mat_entry), sort_rows); // sort by row number
  rows = entries[nz - 1].row + 1; // last item is greatest row (zero indexed)
  if (rows % warp_size) {         // pad group number to warp_size here
    rows += warp_size - rows % warp_size;
  }
  stats = (row_stats *)calloc(rows, sizeof(row_stats)); // set to 0
  *data_row_map = (int *)calloc(rows, sizeof(int));
  irow = entries[0].row; // set first row

  // printf("First row %d\n", irow);
  for (i = 0; i < nz; i++) { // loop through each sorted entry
    if (entries[i].row != irow || i == nz - 1) { // new row
      // printf("%d != %d\n", entries[i].row, irow);
      if (i == nz - 1) {
        // last item, add it to current row
        // printf("Last item i=%d, row=%d, irow=%d\n", i, entries[i].row, irow);
        icol++;
      }
      // hit a new row, record stats for the last row (i-1)
      stats[irow].size = icol;                // record # cols in previous row
      stats[irow].index = entries[i - 1].row; // row # for previous stat item
      // printf("Row %d, i=%d, irow=%d\n", entries[i].row, i, irow);
      stats[irow].start = istart; // starting location in entries array
      // set stats for the next row until this break again
      icol = 0; // reset row items
      irow = entries[i].row;
      istart = i;
    }
    icol++; // keep track of number of items in this row
  }

  *nz_count_len = rows / warp_size + rows % warp_size;
  *nz_count =
      (int *)malloc(*nz_count_len * sizeof(int)); // only one value per group

  /* sort based upon row size, greatest first */
  qsort(stats, rows, sizeof(row_stats), sort_stats);
  /* figure out padding and grouping */
  if (debug_level >= 1) {
    printf("Padding data....%d rows, %d groups\n", rows, *nz_count_len);
  }
  int pad_to, total_padding = 0, pack_to;
  pad_rows *= pack_size; // change padding to account for packed items
  for (i = 0; i < rows; i++) {
    // record JDS to real row number
    (*data_row_map)[i] = stats[i].index;
    if (i < rows - 1) {
      // (*data_row_map)[i]--; // ???? no idea why this is off by 1
    }
    // each row is padded so the number of packed groups % pad_rows == 0
    if (i % warp_size ==
        0) { // on a group boundary with the largest number of items
      // find padding in individual items
      if (stats[i].size % pad_rows) {
        stats[i].padding =
            pad_rows - (stats[i].size % pad_rows); // find padding
      } else {
        stats[i].padding = 0; // no padding necessary, already at pad multiple
      }
      if (stats[i].size % pack_size) {
        pack_to = ceil((float)stats[i].size / pack_size);
      } else {
        pack_to = stats[i].size / pack_size;
      }
      // pack_to = stats[i].size + (!stats[i].size%pack_size) ? 0 : (pack_size -
      // stats[i].size%pack_size);
      pad_to = stats[i].size +
               stats[i].padding; // total size of this row, with padding
      // TODO: change this to reflect the real number of nonzero packed items,
      // not the padded
      // value
      (*nz_count)[i / warp_size] =
          pack_to;                      // number of packed items in this group
      total_size += pad_to * warp_size; // allocate size for this padded group
      if (debug_level >= 2)
        printf("Padding warp group %d to %d items, zn = %d\n", i / warp_size,
               pad_to, pack_to);
    } else {
      stats[i].padding = pad_to - stats[i].size;
    }
    total_padding += stats[i].padding;
    // if (debug_level >= 2)
    //    printf("Row %d, %d items, %d padding\n", stats[i].index,
    //    stats[i].size, stats[i].padding);
  }

  /* allocate data and data_row_index */
  if (debug_level >= 1)
    printf("Allocating data space: %d entries (%f%% padding)\n", total_size,
           (float)100 * total_padding / total_size);
  *data = (float *)calloc(total_size,
                          sizeof(float)); // set to 0 so padded values are set
  *data_col_index =
      (int *)calloc(total_size, sizeof(int)); // any unset indexes point to 0
  *data_row_ptr = (int *)calloc(rows, sizeof(int));
  *len = total_size;
  i = 0; // data index, including padding

  /*
   * Keep looping through each row, writing data a group at a time
   * to the output array. Increment `irow` each time, and use it as
   * an index into entries along with stats.start to get the next
   * data item
   */
  irow = 0;      // keep track of which row we are in inside the fubini-ed array
  int idata = 0; // position within final data array
  int entry_index, j;
  int ipack; // used in internal loop for writing packed values
  mat_entry entry;
  while (1) {
    /* record data_row_ptr */
    (*data_row_ptr)[irow] = idata;

    /* End condtion: the size of the greatest row is smaller than the current
      Fubini-ed row */
    if (stats[0].size + stats[0].padding <= irow * pack_size)
      break;

    // printf("Data row pointer for row %d is %d\n", irow, idata);
    for (i = 0; i < rows; i++) {
      /* take one packed group from each original row */
      // printf("Output irow %d icol %d (real %d,%d size %d)\n", irow, i,
      // entry.col, i, stats[i].size);
      /* Watch out for little vs big endian, and how opencl interprets vector
       * casting from pointers */
      for (ipack = 0; ipack < pack_size; ipack++) {
        if (stats[i].size > irow * pack_size + ipack) {
          // copy value
          entry_index = stats[i].start + irow * pack_size + ipack;
          entry = entries[entry_index];
          /* record index and value */
          (*data)[idata] = entry.val;
          /* each data item will get its row index from the thread, col from
           * here */
          (*data_col_index)[idata] = entry.col;

          if (debug_level >= 2) {
            if (i < 3) {
              // first row debugging
              printf("[%d row%d=%.3f]", ipack + 1, i, entry.val);
            } else {
              printf("%d", ipack + 1);
            }
          }
        } else if (stats[i].size + stats[i].padding >
                   irow * pack_size + ipack) {
          /* add padding to the end of each row here - this assumes padding is
           * factored into allocated size */
          if (debug_level >= 2)
            printf("0");
          (*data_col_index)[idata] = -1;
        } else {
          goto endwrite; // no data written this pass, so don't increment idata
        }
        idata += 1;
      }
    }
  endwrite:
    if (debug_level >= 2) {
      printf("\n");
    }
    irow += 1;
  }

  if (debug_level >= 1)
    printf("Finished converting.\nJDS format has %d columns, %d rows.\n", rows,
           irow);
  free(entries);
  free(stats);
  printf("nz_count_len = %d\n", *nz_count_len);

  *data_cols = rows;
  *data_ptr_len = irow + 1;
  return 0;
}
