/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include "file.h"

unsigned short
read16u(FILE *f)
{
  int n;

  n = fgetc(f);
  n += fgetc(f) << 8;

  return n;
}

short
read16i(FILE *f)
{
  int n;

  n = fgetc(f);
  n += fgetc(f) << 8;

  return n;
}

void
write32u(FILE *f, unsigned int i)
{
  putc(i, f);
  putc(i >> 8, f);
  putc(i >> 16, f);
  putc(i >> 24, f);
}

void
write16u(FILE *f, unsigned short h)
{
  putc(h, f);
  putc(h >> 8, f);
}

void
write16i(FILE *f, short h)
{
  putc(h, f);
  putc(h >> 8, f);
}
