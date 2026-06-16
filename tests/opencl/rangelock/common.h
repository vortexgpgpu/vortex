#ifndef COMMON_H
#define COMMON_H

// Number of consecutive locks each work-item holds SIMULTANEOUSLY (nested).
// This is the depth of coexisting lock-holder contexts a warp must keep live.
// Set WINDOW > the SCS hardware split-table depth K to force the memory-backed
// split spill path (the "bounded-K limitation" case from the SCS proposal).
#define WINDOW 8

#endif // COMMON_H
