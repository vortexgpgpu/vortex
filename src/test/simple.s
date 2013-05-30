/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
********************************************************************************

 Sample HARP assmebly program.

*******************************************************************************/
/* Simple example. */

.align 4096
.perm x
.entry
.global
entry:       ldi %r7, hello
             jali %r5, puts

             trap; /* All traps currently cause a halt. */

.perm rw

hello:
.byte 0x22
.string "Harp!\" is how a harp seal says hello!\n"
