/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <parboil.h>

#include "atom.h"
#include "cutoff.h"

void
write_lattice_summary(const char *filename, Lattice *lattice)
{
  float *lattice_data = lattice->lattice;
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;

  /* Open output file */
  FILE *outfile = fopen(filename, "w");

  if (outfile == NULL) {
    fprintf(stderr, "Cannot open output file\n");
    exit(1);
  }

  /* Write the sum of the the absolute values of all lattice potentials */
  {
    double abspotential = 0.0;
    float tmp;
    int i;

    for (i = 0; i < nx * ny * nz; i++)
      abspotential += fabs((double) lattice_data[i]);

    tmp = (float) abspotential;

    fwrite(&tmp, 1, sizeof(float), outfile);
  }

  /* Write the size of a lattice plane */
  {
    uint32_t tmp;

    tmp = (uint32_t) (lattice->dim.nx * lattice->dim.ny);
    fwrite(&tmp, 1, sizeof(uint32_t), outfile);
  }

  /* Write the plane of lattice data at z=0 and z = nz-1 */
  {
    int plane_size = nx * ny;

    fwrite(lattice_data, plane_size, sizeof(float), outfile);
    fwrite(lattice_data + (nz-1) * plane_size, plane_size, sizeof(float),
	   outfile);
  }

  /* Cleanup */
  fclose(outfile);
}
