// Placeholder device program.
//
// This file used to contain every mode's kernel in one binary. It does not any more:
// each mode is built from its own kernel_m<N>.cpp into its own kernel_m<N>.vxbin, so a
// mode's cycle count cannot depend on which other modes exist in the tree (see
// AGENTS.md -- the combined build moved mode 2 by 56 % when modes 3/4 were added, with a
// byte-identical kernel body).
//
// It stays only because common.mk's kernel.elf rule needs VX_SRCS to name a file. It
// defines no kernel entries; nothing loads kernel.vxbin.
