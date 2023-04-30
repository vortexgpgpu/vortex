//this code tests the ROR intrinsic we added
#include <stdio.h>
#include "vx_intrinsics.h"
#include <stdint.h>
int main(void)
{
        printf("Hello World!\n");
        uint32_t result;
        int x = 8, n=2; 
        int y = (int)((unsigned)x >> n);
        int z = x << (32 - n);
        int g = y | z;
        result = __intrin_rotr_imm(8, 2);
        printf("Result = %ld and verif = %ld\n", result, g);
        return 0;
}


