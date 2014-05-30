.perm x
.entry
.global
entry: ldi %r0, #0x1234;
       ldi %r3, #100;

loop:  ori %r7, %r0, #0;
       jali %r5, printhex
       jali %r5, lfsr_step

       subi %r3, %r3, #1
       rtop @p0, %r3
 @p0 ? jmpi loop
	
       halt

/* %r0: value and return value 
 * %r5: return address
 */
lfsr_step: shri %r1, %r0, #30
           shri %r2, %r0, #2
           xor  %r1, %r1, %r2
           andi %r1, %r1, #1
           shli %r0, %r0, #1
           or %r0, %r0, %r1
           jmpr %r5
