/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* sieve of eratosthanes: Find some primes. */
.def SIZE 0x1000

.align 4096
.perm x
.entry
.global
entry:
             ldi  %r0, #2; /* i = 2 */
loop1:       addi %r7, %r0, #0;
    
             muli %r1, %r0, __WORD;
             st   %r0, %r1, array;
             addi %r0, %r0, #1;
             subi %r1, %r0, SIZE;
             rtop @p0, %r1;
       @p0 ? jmpi loop1;

             ldi  %r0, #1;    
loop2:       addi %r0, %r0, #1;
             muli %r1, %r0, __WORD;
             ld   %r1, %r1, array;
             rtop @p0, %r1;
             notp @p0, @p0;
       @p0 ? jmpi loop2;

             mul   %r2, %r1, %r1;
             subi  %r3, %r2, SIZE;
             neg   %r3, %r3
             isneg @p0, %r3;
       @p0 ? jmpi  end;

             ldi   %r3, #0;
loop3:       muli  %r4, %r2, __WORD;
             st    %r3, %r4, array;
             add   %r2, %r2, %r1;
             ldi   %r4, SIZE;
             sub   %r4, %r2, %r4;
             isneg @p0, %r4;
             notp  @p0, @p0;
       @p0 ? jmpi  loop2;
             jmpi  loop3;

end:         ldi  %r0, __WORD; /* i = 2 */
             shli %r0, %r0, #1;
             ldi  %r11, #0;
loop4:       ld   %r1, %r0, array;
             rtop @p0, %r1;
       @p0 ? addi %r7, %r1, #0;
       @p0 ? jali %r5, printdec;
             rtop @p0, %r1;
       @p0 ? addi %r11, %r11, #1;
             addi %r0, %r0, __WORD;
             ldi %r5, __WORD;
             muli %r5, %r5, SIZE;
             sub  %r1, %r0, %r5;
             rtop @p0, %r1;
       @p0 ? jmpi loop4;

             addi %r7, %r11, #0;
             jali %r5, printdec;
             trap; /* All traps currently cause a halt. */

.perm rw /* TODO: How should I write section permissions? */

.global
array: .space 0x1000 /* SIZE words of space. */
