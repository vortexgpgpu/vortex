/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly programs. These should work on anything from a 4x up.

*******************************************************************************/
/* Library: print decimals and strings! */
.perm x

.global
printhex:    ldi %r8, (__WORD * 8);
             ldi %r11, #1;
             shli %r11, %r11, (__WORD*8 - 1);
printhex_l1: subi %r8, %r8, #4;
             shr %r9, %r7, %r8;
             andi %r9, %r9, #15;
             subi %r10, %r9, #10;
             isneg @p0, %r10;
             notp @p1, @p0;
       @p0 ? addi %r9, %r9, #0x30
       @p1 ? addi %r9, %r10, #0x61
             rtop @p0, %r8;
             st %r9, %r11, #0;
       @p0 ? jmpi printhex_l1;
             ldi %r9, #0x0a;
             st %r9, %r11, #0;
             jmpr %r5;

.global
printfloat:  ori %r10, %r5, #0;
             ori %r11, %r7, #0;
             ftoi %r7, %r7;
             jali %r5, printdecint; /* Print whole part */

             ldi %r8, #1;
             shli %r8, %r8, (__WORD*8 -1 );

             /* Print dot */
             ldi %r7, #0x2e;
             st %r7, %r8, #0;

             ldi %r7, #10;
             itof %r7, %r7;

             ldi %r12, #3

pfloop:      /* Print next decimal place */
             subi %r12, %r12, #1;
             rtop @p0, %r12;
             fmul %r11, %r11, %r7;
             ftoi %r9, %r11;
             modi %r9, %r9, #10;
             addi %r9, %r9, #0x30;
             st %r9, %r8, #0;
       @p0 ? jmpi pfloop;

             ldi %r9, #0x0a;
             st %r9, %r8, #0;

             jmpr %r10;

printdecint: ldi %r8, #1;
             shli %r8, %r8, (__WORD*8 - 1);
             and %r6, %r8, %r7;
             rtop @p0, %r6;
       @p0 ? ldi %r6, #0x2d;
       @p0 ? st %r6, %r8, #0;
       @p0 ? neg %r7, %r7;
             ldi %r9, #0;
printdec_l1: modi %r6, %r7, #10;
             divi %r7, %r7, #10;
             addi %r6, %r6, #0x30;
             st   %r6, %r9, digstack;
             addi %r9, %r9, __WORD;
             rtop @p0, %r7;
       @p0 ? jmpi printdec_l1;
printdec_l2: subi %r9, %r9, __WORD;
             ld %r6, %r9, digstack;
             st %r6, %r8, #0;
             rtop @p0, %r9;
       @p0 ? jmpi printdec_l2;

             jmpr %r5

.global
printdec: ori %r10, %r5, #0;
          jali %r5, printdecint;
          ldi %r6, #0x0a;
          st %r6, %r8, #0;
          jmpr %r10;

.global
puts:        ldi %r8, #1;
             shli %r8, %r8, (__WORD*8 - 1);

puts_l:      ld   %r6, %r7, #0;
             andi %r6, %r6, #0xff;
             rtop @p0, %r6;
             notp @p0, @p0;
       @p0 ? jmpi puts_end;
             st %r6, %r8, #0;
             addi %r7, %r7, #1;
             jmpi puts_l;
puts_end:    jmpr %r5

.global
rand: ldi %r0, #0;
      ld %r0, %r0, rseed;
      mul %r0, %r0, %r0;
      shri %r0, %r0, #8;
      ldi %r9, #0xffff;
      and %r0, %r0, %r9;
      ldi %r9, #0;
      st %r0, %r9, rseed;
      jmpr %r5;

.global
randf: ori %r10, %r5, #0;
       jali %r5, rand;
       ldi %r9, #1;
       shli %r9, %r9, #16;
       itof %r9, %r9;
       itof %r0, %r0;
       fdiv %r0, %r0, %r9;
       jmpr %r10;

.perm rw
rseed:       .word 0xface
digstack:    .space 10
