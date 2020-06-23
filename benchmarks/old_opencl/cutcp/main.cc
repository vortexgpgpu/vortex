/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <parboil.h>

#include "atom.h"
#include "cutoff.h"
#include "output.h"

#define ERRTOL 1e-4f

#define NOKERNELS             0
#define CUTOFF1               1
#define CUTOFF6              32
#define CUTOFF6OVERLAP       64
#define CUTOFFCPU         16384


int appenddata(const char *filename, int size, double time) {
  FILE *fp;
  fp=fopen(filename, "a");
  if (fp == NULL) {
    printf("error appending to file %s..\n", filename);
    return -1;
  }
  fprintf(fp, "%d  %.3f\n", size, time);
  fclose(fp);
  return 0;
}

LatticeDim
lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h)
{
  LatticeDim ret;

  ret.nx = (int) floorf((hi.x-lo.x)/h) + 1;
  ret.ny = (int) floorf((hi.y-lo.y)/h) + 1;
  ret.nz = (int) floorf((hi.z-lo.z)/h) + 1;
  ret.lo = lo;
  ret.h = h;

  return ret;
}

Lattice *
create_lattice(LatticeDim dim)
{
  int size;
  Lattice *lat = (Lattice *)malloc(sizeof(Lattice));

  if (lat == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  lat->dim = dim;

  /* Round up the allocated size to a multiple of 8 */
  size = ((dim.nx * dim.ny * dim.nz) + 7) & ~7;
  lat->lattice = (float *)calloc(size, sizeof(float));

  if (lat->lattice == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  return lat;
}


void
destroy_lattice(Lattice *lat)
{
  if (lat) {
    free(lat->lattice);
    free(lat);
  }
}

int main(int argc, char *argv[]) {
  Atoms *atom;

  LatticeDim lattice_dim;
  Lattice *gpu_lattice;
  Vec3 min_ext, max_ext;	/* Bounding box of atoms */
  Vec3 lo, hi;			/* Bounding box with padding  */

  float h = 0.5f;		/* Lattice spacing */
  float cutoff = 12.f;		/* Cutoff radius */
  float exclcutoff = 1.f;	/* Radius for exclusion */
  float padding = 0.5f;		/* Bounding box padding distance */

  int n;

  struct pb_Parameters *parameters;
  struct pb_TimerSet timers;

  /* Read input parameters */
  parameters = pb_ReadParameters(&argc, argv);
  if (parameters == NULL) {
    exit(1);
  }

  parameters->inpFiles = (char **)malloc(sizeof(char *) * 2);
  parameters->inpFiles[0] = (char *)malloc(100);
  parameters->inpFiles[1] = NULL;
  strncpy(parameters->inpFiles[0], "watbox.sl40.pqr", 100);

  /* Expect one input file */
  if (pb_Parameters_CountInputs(parameters) != 1) {
    fprintf(stderr, "Expecting one input file\n");
    exit(1);
  }

  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  printf("OK\n");

  {
    const char *pqrfilename = parameters->inpFiles[0];

    if (!(atom = read_atom_file(pqrfilename))) {
      fprintf(stderr, "read_atom_file() failed\n");
      exit(1);
    }
    printf("read %d atoms from file '%s'\n", atom->size, pqrfilename);
  }

  printf("OK\n");

  /* find extent of domain */
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  get_atom_extent(&min_ext, &max_ext, atom);
  printf("extent of domain is:\n");
  printf("  minimum %g %g %g\n", min_ext.x, min_ext.y, min_ext.z);
  printf("  maximum %g %g %g\n", max_ext.x, max_ext.y, max_ext.z);

  printf("padding domain by %g Angstroms\n", padding);
  lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
  hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
  printf("domain lengths are %g by %g by %g\n", hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);

  lattice_dim = lattice_from_bounding_box(lo, hi, h);
  gpu_lattice = create_lattice(lattice_dim);
  printf("\n");

  /*
   *  Run OpenCL kernel
   *  (Begin and end with COMPUTE timer active)
   */
  if (gpu_compute_cutoff_potential_lattice(&timers, gpu_lattice, cutoff, atom, 0, parameters)) {
    fprintf(stderr, "Computation failed\n");
    exit(1);
  }

  /*
   * Zero the lattice points that are too close to an atom.  This is
   * necessary for numerical stability.
   */
  if (remove_exclusions(gpu_lattice, exclcutoff, atom)) {
    fprintf(stderr, "remove_exclusions() failed for gpu lattice\n");
    exit(1);
  }

  printf("\n");

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  /* Print output */
  if (parameters->outFile) {
    //write_lattice_summary(parameters->outFile, gpu_lattice);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Cleanup */
  destroy_lattice(gpu_lattice);
  free_atom(atom);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
