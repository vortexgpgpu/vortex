#!/bin/bash
# Detect startup features needed from a linked ELF binary.
# Usage: kernel_startup.sh <objdump> <elf-file>
# Outputs preprocessor flags for vx_start.S compilation.

OBJDUMP=$1
ELF=$2

sections=$($OBJDUMP -h "$ELF" 2>/dev/null)

if echo "$sections" | awk '$2 ~ /^\.(s?data|s?bss)$/ && $3 !~ /^0+$/ { found=1; exit } END { exit !found }'; then
  printf " -DNEED_GP"
fi

if echo "$sections" | awk '$2 ~ /^\.(tdata|tbss)$/ && $3 !~ /^0+$/ { found=1; exit } END { exit !found }'; then
  printf " -DNEED_TLS"
fi

if echo "$sections" | awk '$2 ~ /^\.(preinit_array|init_array)$/ && $3 !~ /^0+$/ { found=1; exit } END { exit !found }'; then
  printf " -DNEED_INITFINI"
fi
