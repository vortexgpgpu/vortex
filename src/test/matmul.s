/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* Matrix multiply: find matrix product */
.def SIZE 0x1000

.align 4096
.perm x
.entry
.global
entry: ldi %r0, matrix_a;
       ldi %r1, matrix_b;
       ldi %r2, matrix_r;
       ldi %r3, #3;
       jali %r5, matmul;

       ldi %r0, #64;
       ldi %r1, matrix_r;
ploop: ld %r7, %r1, #0;
       jali %r5, printfloat;
       subi %r0, %r0, #1;
       addi %r1, %r1, __WORD;
       rtop @p0, %r0;
 @p0 ? jmpi ploop;

       trap;

/* Write the matrix product of square matrix at (%r0) and (%r1) to (%r2). The
   size of these matrices is 2^Nx2^N, where N = %r3 */
matmul: ldi %r4, #1;
        shl %r4, %r4, %r2;

        ldi %r9, #0; /* result row: %r9 */
rloop:  

        ldi %r6, #0; /* result col: %r6 */
cloop:  

        ldi %r8, #0; /* dot prod position: %r8 */
iloop:  /* TODO : dot product */

        addi %r8, %r8, #1;
        sub %r7, %r8, %r4;
        rtop @p0, %r7;
  @p0 ? jmpi iloop;

        /* TODO : store result of dot product */

        addi %r6, %r6, #1;
        sub %r7, %r6, %r4;
        rtop @p0, %r7;
  @p0 ? jmpi cloop;

        addi %r9, %r9, #1;
        sub %r7, %r9, %r4;
        rtop @p0, %r7;
  @p0 ? jmpi rloop;

        jmpr %r5;

.align 4096
.perm rw
matrix_a: .word  1f  2f  3f  4f  5f  6f  7f  8f
          .word  2f  3f  4f  5f  6f  7f  8f  9f
          .word  3f  4f  5f  6f  7f  8f  9f 10f
          .word  4f  5f  6f  7f  8f  9f 10f 11f
          .word  5f  6f  7f  8f  9f 10f 11f 12f
          .word  6f  7f  8f  9f 10f 11f 12f 13f
          .word  7f  8f  9f 10f 11f 12f 13f 14f
          .word  8f  9f 10f 11f 12f 13f 14f 15f

matrix_b: .word  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
          .word  1.0 1.1 1.2 1.3 1.4 1.5 1.7 1.7
          .word  2.0 2.1 2.2 2.3 2.4 2.5 2.7 2.7
          .word  3.0 3.1 3.2 3.3 3.4 3.5 3.7 3.7
          .word  4.0 4.1 4.2 4.3 4.4 4.5 4.7 4.7
          .word  5.0 5.1 5.2 5.3 5.4 5.5 5.7 5.7
          .word  6.0 6.1 6.2 6.3 6.4 6.5 6.7 6.7
          .word  7.0 7.1 7.2 7.3 7.4 7.5 7.7 7.7

matrix_r: .space 64

retaddr: .word 0
