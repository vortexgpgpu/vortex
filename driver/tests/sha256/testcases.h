#ifndef TESTCASES_H
#define TESTCASES_H

#include "sha256.h"

#define MAX_DATA_SIZE 256

typedef struct {
    uint32_t msgsize;
    char msg[MAX_DATA_SIZE];
    char expected_digest[DIGEST_BYTES];
} testcase_t;

#define DEFAULT_TESTNO 1
#define N_TESTCASES 3
extern testcase_t testcases[N_TESTCASES];

#endif
