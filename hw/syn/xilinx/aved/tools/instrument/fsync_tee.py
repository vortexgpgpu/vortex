#!/usr/bin/env python3
"""A tee that fsync()s every line.

Why this exists: the 2026-08-24 13:44 ladder run produced a **0-byte** log.
`tee` buffers, the host hard-reset a few seconds later, and everything the test
had printed was lost -- including the output of the rung that was executing.
The fsync'd breadcrumb survived and the log did not, which is the whole lesson.

  <command> 2>&1 | python3 fsync_tee.py <logfile>

Copies stdin to stdout and to <logfile>, flushing and fsyncing after every
line, so whatever reached this process is on stable storage before the next
line is read.
"""
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: fsync_tee.py <logfile>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with open(path, "ab", buffering=0) as fh:
        for line in sys.stdin.buffer:
            try:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                pass
            fh.write(line)
            os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
