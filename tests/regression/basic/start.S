.section .init, "ax"
.global _start
.type _start, @function
_start:
  # call main routine
  call main

  # dump perf counter
  call vx_perf_dump

  # end execution
  .insn r 0x0b, 0, 0, x0, x0, x0
.size _start, .-_start