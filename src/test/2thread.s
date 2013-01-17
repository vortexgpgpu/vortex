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
Array1:
.word -1 -2 -3 -4 -5 -6 -7 -8

Array2:
.word  1  2  3  4  5  6  7  8
