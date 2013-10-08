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
        ldi %r10, (`__WORD); /* ` is the log base 2 operator */
        shl %r4, %r4, %r3;
        add %r10, %r10, %r3;
        ldi %r14, #1;
        shl %r14, %r14, %r10;

        ldi %r9, #0; /* result row: %r9 */
rloop:  ldi %r6, #0; /* result col: %r6 */
cloop:  shli %r16, %r6, (`__WORD);
        shl %r15, %r9, %r10;

        add %r11, %r15, %r0;
        add %r12, %r16, %r1;
        ldi %r13, #0;

        ldi %r8, #0 /* dot prod position: %r8 */
iloop:  ld %r7, %r11, #0;
        ld %r17, %r12, #0;
        fmul %r7, %r7, %r17
        fadd %r13, %r13, %r7;

        addi %r8, %r8, #1;
        addi %r11, %r11, __WORD;
        add %r12, %r12, %r14;
        sub %r7, %r8, %r4;
        rtop @p0, %r7;
  @p0 ? jmpi iloop;

        add %r15, %r15, %r16;
        add %r15, %r15, %r2;
        st %r13, %r15, #0;

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
          .word  1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7
          .word  2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7
          .word  3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7
          .word  4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7
          .word  5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7
          .word  6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7
          .word  7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7

matrix_r: .space 64

retaddr: .word 0
