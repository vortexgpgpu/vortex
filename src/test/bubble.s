/* Bubble Sort */

.def SIZE 100

.align 4096
.perm x
.global

entry:  ldi %r1, Array;          /* p = array;               */
        ldi %r6, ArrayEnd;       /* lim = array+N */

oloop:                           /* do {                     */
        ldi %r2, Array;          /*   q = array;             */
        ldi %r7, #1;             /*   sorted = 1;            */
iloop:
        sub %r3, %r2, %r6;       /*   while (q != lim) {     */
        iszero @p0, %r3;
  @p0 ? jmpi onext;
        
        ld %r3, %r2, #0;         /*     if (*q > *(q + 1)) { */
        ld %r4, %r2, __WORD;     /*       temp = *(q + 1);   */
        sub %r5, %r4, %r3;
        isneg @p0, %r5;
        notp @p0, @p0;
  @p0 ? jmpi inext;
        st %r3, %r2, __WORD;     /*       *(q + 1) = *q;     */
        st %r4, %r2, #0;         /*       *q = temp;         */
        ldi %r7, #0;             /*       sorted = 0;        */
                                 /*     }                    */
inext:  addi %r2, %r2, __WORD;   /*     q++;                 */
        jmpi iloop;              /*   }                      */
onext:  addi %r1, %r1, __WORD;   /*   p++;                   */
        subi %r6, %r6, __WORD;   /*   lim--;                 */
        rtop @p0, %r7;           /* } while (!sorted && p != array+N)   */
  @p0 ? jmpi printresults;
        subi %r3, %r1, ArrayEnd;
        rtop @p0, %r3;
  @p0 ? jmpi oloop;

/* p = array; do { printdec(*p); p++; } while(p != array+N); */
printresults: ldi %r1, Array;
printloop:    ld %r7, %r1, #0;
              jali %r5, printdec;
              addi %r1, %r1, __WORD;
              subi %r2, %r1, ArrayEnd;
              rtop @p0, %r2;
        @p0 ? jmpi printloop;
              trap; 

.align 4096
.perm rw
Array:
  .word 3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6 4 3 3 8 3 2 7 9 5 0
  .word 2 8 8 4 1 9 7 1 6 9 3 9 9 3 7 5 1 0 5 8 2 0 9 7 4 9 4 4 5 9 2 3 0
  .word 7 8 1 6 4 0 6 2 8 6 2 0 8 9 9 8 6 2 8 0 3 4 8 2 5 3 4 2 1 1 7 0 6 7
ArrayEnd: .word 0
