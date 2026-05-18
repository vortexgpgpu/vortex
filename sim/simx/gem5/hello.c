// Phase 0 ARM SE-mode smoke test. Cross-compile with
//   aarch64-linux-gnu-gcc -static -o /tmp/hello-arm hello.c
// and run under gem5 with the new gem5_library SimpleBoard wiring
// (or the deprecated configs/example/se.py if still available).
// Confirms the cross-toolchain produces something gem5 can load.

#include <stdio.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    printf("Hello, ARM SE-mode (gem5 v25 Phase 0)\n");
    return 0;
}
