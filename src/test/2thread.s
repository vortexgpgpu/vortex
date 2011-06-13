/* 2-SIMD-thread test program! */

.perm x
.entry
.global
entry: ldi %r0, #0;

      ldi %r2, #1;
      ldi %r1, Array2;
      clone %r2;
      ldi %r1, Array1;
      ldi %r5, #2;
      jalis %r5, %r5, sumArr;
      ldi %r7, Array1;
      ld %r7, %r7, #0;
      jali %r5, printdec;
      ldi %r7, Array2;;
      ld %r7, %r7, #0;
      jali %r5, printdec
      trap;

/* Sum multiple arrays at once through the magic of SIMD! */
sumArr: ldi %r3, #0;
        ldi %r4, #8;
loop:   ld  %r2, %r1, #0;
        add %r3, %r3, %r2;
        addi %r1, %r1, #8;
        subi %r4, %r4, #1;
        rtop @p0, %r4;
  @p0 ? jmpi loop;
        st %r3, %r1, #-64;
        jmprt %r5;

.perm rw
.word Array1    -1
.word Array1_01 -2
.word Array1_02 -3
.word Array1_03 -4
.word Array1_04 -5
.word Array1_05 -6
.word Array1_06 -7
.word Array1_07 -8

.word Array2    1
.word Array2_00 2
.word Array2_01 3
.word Array2_02 4
.word Array2_03 5
.word Array2_04 6
.word Array2_05 7
.word Array2_06 8
