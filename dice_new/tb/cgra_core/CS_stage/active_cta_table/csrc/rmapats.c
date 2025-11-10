// file = 0; split type = patterns; threshold = 100000; total count = 0.
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "rmapats.h"

void  schedNewEvent (struct dummyq_struct * I1473, EBLK  * I1468, U  I619);
void  schedNewEvent (struct dummyq_struct * I1473, EBLK  * I1468, U  I619)
{
    U  I1763;
    U  I1764;
    U  I1765;
    struct futq * I1766;
    struct dummyq_struct * pQ = I1473;
    I1763 = ((U )vcs_clocks) + I619;
    I1765 = I1763 & ((1 << fHashTableSize) - 1);
    I1468->I665 = (EBLK  *)(-1);
    I1468->I666 = I1763;
    if (0 && rmaProfEvtProp) {
        vcs_simpSetEBlkEvtID(I1468);
    }
    if (I1763 < (U )vcs_clocks) {
        I1764 = ((U  *)&vcs_clocks)[1];
        sched_millenium(pQ, I1468, I1764 + 1, I1763);
    }
    else if ((peblkFutQ1Head != ((void *)0)) && (I619 == 1)) {
        I1468->I668 = (struct eblk *)peblkFutQ1Tail;
        peblkFutQ1Tail->I665 = I1468;
        peblkFutQ1Tail = I1468;
    }
    else if ((I1766 = pQ->I1370[I1765].I688)) {
        I1468->I668 = (struct eblk *)I1766->I686;
        I1766->I686->I665 = (RP )I1468;
        I1766->I686 = (RmaEblk  *)I1468;
    }
    else {
        sched_hsopt(pQ, I1468, I1763);
    }
}
#ifdef __cplusplus
extern "C" {
#endif
void SinitHsimPats(void);
#ifdef __cplusplus
}
#endif
