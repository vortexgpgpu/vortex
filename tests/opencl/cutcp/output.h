/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef OUTPUT_H
#define OUTPUT_H

#include "cutoff.h"

#ifdef __cplusplus
extern "C" {
#endif

void
write_lattice_summary(const char *filename, Lattice *lattice);

#ifdef __cplusplus
}
#endif

#endif
